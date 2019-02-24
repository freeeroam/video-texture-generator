#include "vtg.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <cmath>
#include <sstream>
#include <boost/filesystem.hpp>
#include <iomanip>

enum similarity_measure sim_measure = rgb;
Video * input_video = nullptr;
bool equalise_brightness = false;
bool stabilise = false;
int tap_filter = 0;
bool anticipate_future_cost = true;
double multiple_trans_tradeoff = 0.95;
double relative_weight_future_trans = 0.99;
int min_matrix_display_size = 500;
bool prune_transitions = true;
bool enable_blending = true;
int num_frames_to_blend = 2;
std::vector <float> * alpha_weights = nullptr;
std::string output_file_path =
  "/home/ant/development/video-texture-generator/textures/";
std::string output_file_name = "output.avi";
std::string parameters_file = "";
std::list <std::string> test_parameters;

extern bool stabilise_video(cv::VideoCapture & cap);

int main(int argc, char ** argv)
{
  if (!check_flags(argc, argv))
  {
    return 0;
  } // if
  output_file_path += output_file_name;

  input_video = load_video(argv[1]);
  if (input_video == nullptr)
  {
    std::cout << "Input file provided is not a valid video." << std::endl;
    return 0;
  } // if

  // Save distance matrix and matrix distance values if not already saved
  boost::filesystem::path distances_txt (output_file_path + "/distances.txt");
  std::vector <std::vector <double>> * distances =
    boost::filesystem::exists(distances_txt)
    ? load_distance_matrix(output_file_path + "/distances.txt")
    : distance_matrix(*input_video);
  if (!boost::filesystem::exists(distances_txt))
  {
    save_distance_values(*distances, output_file_path + "/distances.txt");
  } // if
  boost::filesystem::path distances_img (output_file_path + "/distances.png");
  if (!boost::filesystem::exists(distances_img))
  {
    save_matrix(*distances, output_file_path + "/distances.png");
  } // if

  // Get sets of program parameters from files or from program arguments
  if (parameters_file != "")
  {
    test_parameters = load_parameters_from_file("/home/ant/development/video-texture-generator/"
                              + parameters_file);
  } else
  {
    std::ostringstream params;
    params << std::setprecision(20)
      << "r=" << std::to_string((int)sim_measure)
      << ",b=" << std::to_string((int)equalise_brightness)
      << ",f=" << std::to_string(tap_filter)
      << ",t=" << std::to_string(multiple_trans_tradeoff)
      << ",w=" << std::to_string(relative_weight_future_trans)
      << ",c=" << std::to_string(num_frames_to_blend)
      << '\n';
    test_parameters.push_back(params.str());
  } // else

  // Run tests for all sets of parameters
  std::cout << "Running " << test_parameters.size() << " tests" << std::endl; // debugging
  std::list <std::string> ::const_iterator test_it;
  std::vector <std::vector <double>> * pd_distances = nullptr;
  std::vector <std::vector <double>> * anticipated_distances = nullptr;
  std::vector <std::vector<struct CompoundLoop>> * loop_table = nullptr;
  std::list <struct Transition> * best_transitions = nullptr;
  std::vector <double> * weights = nullptr;
  struct PreRenderedSequence * sequence = nullptr;
  struct CompoundLoop compound_loop;
  std::vector <cv::Mat> * output_frames = nullptr;
  for (test_it = test_parameters.begin(); test_it != test_parameters.end();
       test_it++)
  {
    load_parameters_from_string(*test_it);
    std::cout << "Output will be saved to: " << output_file_path << "/"
              << output_name() << std::endl; // output for debugging

    apply_preprocessing(*input_video);
    if (tap_filter > 0)
    {
      weights = filter_weights(tap_filter * 2);
      pd_distances = preserve_dynamics2(*distances, *weights);
      delete weights;
    } // if

    if (anticipate_future_cost)
    {
      anticipated_distances = anticipate_future(*pd_distances);
      delete pd_distances;
    } // if

    if (prune_transitions)
    {
      select_only_local_maxima2(*anticipated_distances);
      best_transitions = lowest_average_cost_transitions(
        *anticipated_distances, 40);
      delete anticipated_distances;
    } // if

    boost::filesystem::create_directory(output_file_path + "/" + output_name());

    loop_table = dp_table(*best_transitions, 50);
    int videos_produced = 0;
    for (int row_index = loop_table->size() - 1; row_index > 0; row_index--)
    {
      for (int col_index = 0; col_index < best_transitions->size(); col_index++)
      {
        if ((*loop_table)[row_index][col_index].total_cost > 0
            && compound_loop_length((*loop_table)[row_index][col_index]) > 2)
        {
          boost::filesystem::create_directory(output_file_path + "/"
                                              + output_name() + "/"
                                              + std::to_string(videos_produced) + "l"
                                              + std::to_string(row_index));

          compound_loop.transitions = (*loop_table)[row_index][col_index].transitions;
          compound_loop.total_cost = (*loop_table)[row_index][col_index].total_cost;
          schedule_transitions(compound_loop);
          sequence = create_loop_frame_sequence(compound_loop,
                                                output_file_path + "/"
                                                + output_name() + "/"
                                                + std::to_string(
                                                    videos_produced) + "l"
                                                + std::to_string(row_index)
                                                + "/transitions.txt");

          // Blend frames if blending is enabled
          if (enable_blending)
          {
            output_frames = blend_frames_at_transitions(*(sequence->frames),
                                                        sequence->transitions, 4);
          } else
          {
            output_frames = sequence->frames;
          } // else

          // Produce output video
          Video output_video (output_frames, input_video->get_fps(),
                              input_video->get_frame_width(),
                              input_video->get_frame_height());

          // Write output and data
          save_parameters(output_file_path + "/" + output_name()
                          + output_name() + "/"
                          + std::to_string(videos_produced)
                          + "l" + std::to_string(row_index)
                          + "/parameters.txt");
          boost::filesystem::create_directory(output_file_path + "/"
                                              + output_name() + "/"
                                              + std::to_string(videos_produced)
                                              + "l" + std::to_string(row_index)
                                              + "/transition_pairs");
          save_transition_pairs(*(sequence->frames), sequence->transitions,
                                output_file_path + "/" + output_name()
                                + "/" + std::to_string(videos_produced) + "l"
                                + std::to_string(row_index) + "/transition_pairs");
          write_video(output_video, output_file_path + "/" + output_name()
                      + "/" + std::to_string(videos_produced) + "l"
                      + std::to_string(row_index) + "/texture.avi");
          std::cout << std::endl;
          videos_produced++;
          delete sequence;
          delete alpha_weights;
        } // if
      } // for
    } // for

    std::cout << "Test complete *\n" << std::endl;
    delete best_transitions;
    delete loop_table;
  } // for

  /**
  std::vector <std::vector <double>> * probabilities =
    probability_matrix(*distances, 3 * average_distance(*distances));
  normalise_probabilities(*probabilities);
  //display_transition_matrix(*probabilities);
  display_distance_matrix(*distances);
  **/

  delete input_video;
  return 0;
} // function main

// Check command line arguments and returns true if command line arguments
// are not used correctly
bool check_flags(int argc, char ** argv)
{
  if (argc < 2)
  {
    std::cout << "No input file provided" << std::endl;
    std::cout << "Correct usage: ./vtg filepath" << std::endl;
    return false;
  } else if (strcmp(argv[1], "--help") == 0)
  {
    display_help();
    return false;
  } // else if

  for (int arg_index = 2; arg_index < argc; arg_index++)
  {
    if (strlen(argv[arg_index]) > 1
        && argv[arg_index][0] == '-'
        && argv[arg_index][1] == '-')
    {
      std::cout << "Incorrect use of arguments" << std::endl;
      return false;
    } else if (strlen(argv[arg_index]) > 1
               && argv[arg_index][0] == '-')
    {
      for (int char_index = 1; char_index < strlen(argv[arg_index]);
           char_index++)
      {
        if (argv[arg_index][char_index] == 'l') // use luminance as measure
        {
          sim_measure = luminance;
        } else if (argv[arg_index][char_index] == 'o'
                   && argc > arg_index)
        {
          output_file_name = argv[arg_index + 1];
          arg_index++;
          break;
        } else if (argv[arg_index][char_index] == 'i' // parameter input file
                   && argc > arg_index)
        {
          parameters_file = argv[arg_index + 1];
          arg_index++;
          break;
        } else if (argv[arg_index][char_index] == 'b') // equalise brightness
        {
          equalise_brightness = true;
        } else if (argv[arg_index][char_index] == 's') // stabilise first
        {
          stabilise = true;
        } else if (argv[arg_index][char_index] == 'p')
        {
          prune_transitions = true;
        } else if (argv[arg_index][char_index] == 'm'
                   && char_index == strlen(argv[arg_index]) - 1
                   && argc > arg_index) // set matrix display size
        {
          arg_index++;
          try
          {
            if (std::stoi(argv[arg_index]) > 0)
            {
              min_matrix_display_size = std::stoi(argv[arg_index]);
            } else
            {
              std::cout << "Incorrect use of arguments" << std::endl;
              std::cout << "Matrix size must be a positive integer"
                        << std::endl;
            } // else
          } catch (...)
          {
            std::cout << "Incorrect use of arguments" << std::endl;
            return false;
          } // catch
        } else if (argv[arg_index][char_index] == 'f'
                   && char_index == strlen(argv[arg_index]) - 1
                   && argc > arg_index) // set tap filter number
        {
          arg_index++;
          try
          {
            if (std::stoi(argv[arg_index]) > 0)
            {
              tap_filter = std::stoi(argv[arg_index]);
            } else
            {
              std::cout << "Incorrect use of arguments" << std::endl;
              std::cout << "Tap- number for filter must be a positive integer"
                        << std::endl;
              return false;
            } // else
          } catch (...)
          {
            std::cout << "Incorrect use of arguments" << std::endl;
            return false;
          } // catch
        } else if (argv[arg_index][char_index] == 'a') // anticipate future
        {
          anticipate_future_cost = true;
        } else if (argv[arg_index][char_index] == 'w'
                   && char_index == strlen(argv[arg_index]) - 1
                   && argc > arg_index) // set relative weight
        {
          arg_index++;
          try
          {
            if (s_to_doub(argv[arg_index]) > 0
                && s_to_doub(argv[arg_index]) < 1)
            {
              relative_weight_future_trans = s_to_doub(argv[arg_index]);
            } else
            {
              std::cout << "Incorrect use of arguments" << std::endl
                        << "Relative weight of future transitions must be "
                        << "a numerical value between 0 and 1"
                        << std::endl;
              return false;
            } // else
          } catch (...)
          {
            std::cout << "Incorrect use of arguments" << std::endl;
            return false;
          } // catch
        } else if (argv[arg_index][char_index] == 't'
                   && char_index == strlen(argv[arg_index]) - 1
                   && argc > arg_index) // set tradeoff between taking multiple
        { // good low-cost transitions versus a single, poorer one
          arg_index++;
          try
          {
            if (s_to_doub(argv[arg_index]) > 0)
            {
              multiple_trans_tradeoff = s_to_doub(argv[arg_index]);
            } else
            {
              std::cout << "Incorrect use of arguments" << std::endl
                        << "Tradeoff between taking multiple low-cost "
                        << "transitions vs. a single poorer one must be "
                        << "a numerical value greater than 0" << std::endl;
              return false;
            } // else
          } catch (...)
          {
            std::cout << "Incorrect use of arguments" << std::endl;
            return false;
          } // catch
        } else if (argv[arg_index][char_index] == 'c'
                   && char_index == strlen(argv[arg_index]) - 1
                   && argc > arg_index) // set number of frames to use in a
        { // crossfade blend at each transition
          arg_index++;
          try
          {
            if (s_to_doub(argv[arg_index]) > 1
                && s_to_doub(argv[arg_index]) < 7)
            {
              enable_blending = true;
              num_frames_to_blend = s_to_doub(argv[arg_index]);
            } else
            {
              std::cout << "Incorrect use of arguments" << std::endl
                        << "Number of frames in a crossfade blend "
                        << "must be an integer from 2 to 6" << std::endl;
              return false;
            } // else
          } catch (...)
          {
            std::cout << "Incorrect use of arguments" << std::endl;
            return false;
          } // catch
        } // else if
      } // for
    } else
    {
      std::cout << "Incorrect use of arguments" << std::endl;
      return false;
    } // else
  } // for
  return true;
} // function check_flags

// Applies enabled preprocessing methods to the given video
void apply_preprocessing(Video & video)
{
  if (equalise_brightness)
  {
    cv::Mat reference_image = find_reference_image(*input_video, 4);
    equalise_video_brightness(video, reference_image);
  } // if
} // function apply_preprocessing

// Takes a similarity measure and a video.
// Returns a pointer to a distance matrix of the frames in the video.
std::vector <std::vector <double>> * distance_matrix(Video & video)
{
  std::vector <std::vector <double>> * matrix =
    create_square_matrix <double> (video.get_frames().size(), 0);

  std::cout << "Analysing frames..." << std::endl;

  double max_distance = 0;
  // Iterate through matrix and calculate distances between frames
  for (int row_index = 0; row_index < matrix->size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix->size(); col_index++)
    {
      if (row_index == col_index)
      {
        (*matrix)[row_index][col_index] = 0;
      } else
      {
        (*matrix)[row_index][col_index] =
          calculate_distance(video.get_frames()[row_index + 1],
                             video.get_frames()[col_index]);
        if ((*matrix)[row_index][col_index] > max_distance)
        {
          max_distance = (*matrix)[row_index][col_index];
        } // if
      } // else
    } // for
  } // for

  // Change invalid distances of "-1" to the maximum distance
  for (int row_index = 0; row_index < matrix->size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix->size(); col_index++)
    {
      if ((*matrix)[row_index][col_index] == -1)
      {
        (*matrix)[row_index][col_index] = max_distance;
      } // if
    } // for
  } // for
  return matrix;
} // function distance_matrix

// Allocates and initialises a square matrix of given size.
// Returns a pointer to this matrix.
template <class T>
std::vector <std::vector <T>> * create_square_matrix(unsigned int size,
                                                     T initial_value)
{
  std::vector <std::vector <T>> * matrix =
    new std::vector <std::vector <T>> (size, std::vector <T> (size,
                                                              initial_value));
  return matrix;
} // function create_square_matrix

// Calculates and returns the L2 difference between two frames.
double calculate_distance(cv::Mat & frame_a, cv::Mat & frame_b)
{
  if (frame_a.rows != frame_b.rows
      || frame_a.cols != frame_b.cols
      || frame_a.channels() != frame_b.channels()
      || frame_a.isContinuous() != frame_b.isContinuous())
  {
    return -1;
  } // if

  int rows = frame_a.rows;
  int cols = frame_a.cols;
  int channels = frame_a.channels();
  if (frame_a.isContinuous())
  {
    cols *= rows;
    rows = 1;
  } // if

  double total = 0;
  uchar * row_a, * row_b;
  double val_a, val_b;
  for (int row_index = 0; row_index < rows; row_index++)
  {
    row_a = frame_a.ptr <uchar> (row_index);
    row_b = frame_b.ptr <uchar> (row_index);
    for (int pix_index = 0; pix_index < cols * channels;
         pix_index += 3)
    {
      if (sim_measure == luminance)
      {
        val_a = 0.299 * row_a[pix_index + 2]
          + 0.587 * row_a[pix_index + 1]
          + 0.114 * row_a[pix_index];
        val_b = 0.299 * row_b[pix_index + 2]
          + 0.587 * row_b[pix_index + 1]
          + 0.114 * row_b[pix_index];
        total += std::pow(val_a - val_b, 2);
      } else
      {
        total += std::pow(row_a[pix_index] - row_b[pix_index], 2)
          + std::pow(row_a[pix_index + 1] - row_b[pix_index + 1], 2)
          + std::pow(row_a[pix_index + 2] - row_b[pix_index + 2], 2);
      } // else
    } // for
  } // for
  return total;
} // function calculate_distance

// Given a distance matrix, produces and returns a matrix representing
// probability of transitioning from one frame to another
std::vector <std::vector <double>> * probability_matrix(
  std::vector <std::vector <double>> & distance_matrix, double sigma)
{
  std::vector <std::vector <double>> * prob_matrix =
    create_square_matrix <double> (distance_matrix.size(), 0);

  std::cout << "Calculating probabilities of transitions..." << std::endl; // Output for debug
  for (int row_index = 0; row_index < distance_matrix.size() - 1; row_index++)
  {
    for (int col_index = 0; col_index < distance_matrix.size();
         col_index++)
    {
      (*prob_matrix)[row_index][col_index] =
        std::exp(-(distance_matrix[row_index + 1][col_index] / sigma));
    } // for
  } // for
  return prob_matrix;
} // function probability_matrix

// Given a distance matrix, calculates and returns the average distance.
double average_distance(std::vector <std::vector <double>> & matrix)
{
  double total = 0;
  int number_nonzeros = 0;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix.size(); col_index++)
    {
      if (matrix[row_index][col_index] != 0)
      {
        total += matrix[row_index][col_index];
        number_nonzeros++;
      } // if
    } // for
  } // for
  return total / number_nonzeros;
} // function average_distance

// Creates a binary image by thresholding the given matrix and then
// displays it in a new window.
void display_binary_matrix(std::vector <std::vector <double>> & matrix)
{
  std::cout << "Displaying matrix..." << std::endl; // output for debugging
  cv::Mat * binary_mat = create_binary_matrix(matrix,
                                              calculate_threshold(matrix));

  int component_length = 1;
  while (component_length * matrix.size() < min_matrix_display_size)
  {
    component_length++;
  } // while

  cv::Mat image (matrix.size() * component_length,
                 matrix.size() * component_length,
                 CV_8U);
  uchar * old_row = nullptr;
  uchar * new_row = nullptr;
  for (int row_index = 0; row_index < matrix.size() * component_length;
       row_index++)
  {
    new_row = image.ptr <uchar> (row_index);
    for (int pix_index; pix_index < matrix.size() * component_length;
         pix_index++)
    {
      new_row[pix_index] =
        binary_mat->at <uchar> (row_index / component_length,
                                pix_index / component_length);
    } // for
  } // for

  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int pix_index = 0; pix_index < matrix.size(); pix_index++)
    {
      for (int com_row_index = 0; com_row_index < component_length;
           com_row_index++)
      {
        for (int com_col_index = 0; com_col_index < component_length;
             com_col_index++)
        {
          image.at <uchar> (row_index * component_length + com_row_index,
                            pix_index * component_length + com_col_index) =
            binary_mat->at <uchar> (row_index, pix_index);
        } // for
      } // for
    } // for
  } // for

  cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Display", on_event, nullptr);
  cv::imshow("Display", image);
  cv::waitKey(0);
} // function display_binary_matrix

// Enlarges the given square matrix to a size greater than the given global
// minimum
cv::Mat * enlarge_matrix(cv::Mat & matrix)
{
  int component_length = 1;
  while (component_length * matrix.rows < min_matrix_display_size)
  {
    component_length++;
  } // while

  cv::Mat * image = new cv::Mat(matrix.rows * component_length,
                                matrix.cols * component_length,
                                CV_8U);
  for (int row_index = 0; row_index < matrix.rows; row_index++)
  {
    for (int pix_index = 0; pix_index < matrix.cols; pix_index++)
    {
      for (int com_row_index = 0; com_row_index < component_length;
           com_row_index++)
      {
        for (int com_col_index = 0; com_col_index < component_length;
             com_col_index++)
        {
          image->at <uchar> (row_index * component_length + com_row_index,
                             pix_index * component_length + com_col_index) =
            matrix.at <uchar> (row_index, pix_index);
        } // for
      } // for
    } // for
  } // for
  return image;
} // function enlarge_matrix

// Displays a heat map representation of the given distance matrix
void display_distance_matrix(std::vector <std::vector <double>> & dist_matrix)
{
  std::cout << "Displaying distance matrix..." << std::endl;

  cv::Mat * image = heat_map(dist_matrix);
  cv::Mat * enlarged_image = enlarge_matrix(*image);
  cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Display", on_event, nullptr);
  cv::imshow("Display", *enlarged_image);
  cv::waitKey(0);
  delete image;
  delete enlarged_image;
} // function display_distance_matrix

// Displays a heat map representation of a given stochastic/probability matrix
void display_transition_matrix(
  std::vector <std::vector <double>> & prob_matrix)
{
  std::cout << "Displaying transition matrix..." << std::endl;

  double maximum_prob = maximum_component_value <double> (prob_matrix);
  double minimum_prob = minimum_component_value <double> (prob_matrix, false);

  cv::Mat image (prob_matrix.size(), prob_matrix.size(), CV_8U,
                 cv::Scalar::all(0));
  uchar * image_row = nullptr;
  for (int row_index = 0; row_index < prob_matrix.size(); row_index++)
  {
    image_row = image.ptr <uchar> (row_index);
    for (int col_index = 0; col_index < prob_matrix.size(); col_index++)
    {
      image_row[col_index] = 255
        * ((prob_matrix[row_index][col_index] - minimum_prob)
           / (maximum_prob - minimum_prob));
    } // for
  } // for

  cv::Mat * output = enlarge_matrix(image);
  cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Display", on_event, nullptr);
  cv::imshow("Display", *output);
  cv::waitKey(0);
} // function display_transition matrix

// Returns a "binary" Mat object with components set to 0 if their
// corresponding components in the original matrix are greater than the given
// threshold, or set to 255 otherwise.
cv::Mat * create_binary_matrix(std::vector <std::vector <double>> & matrix,
                               double threshold)
{
  cv::Mat * output = new cv::Mat(matrix.size(), matrix.size(), CV_8U,
                                 cv::Scalar::all(255));
  uchar * output_row = nullptr;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    output_row = output->ptr <uchar> (row_index);
    for (int col_index = 0; col_index < matrix.size(); col_index++)
    {
      if (matrix[row_index][col_index] > threshold)
      {
        output_row[col_index] = 0;
      } // if
    } // for
  } // for
  return output;
} // function create_binary_matrix

// Returns a "heat map" representation of a given probability matrix.
cv::Mat * heat_map(std::vector <std::vector <double>> & dist_matrix)
{
  cv::Mat * heat_map = new cv::Mat(dist_matrix.size(), dist_matrix.size(),
                                   CV_8U, cv::Scalar::all(255));
  double max_distance = maximum_component_value <double> (dist_matrix);
  double min_distance = minimum_component_value <double> (dist_matrix, false);

  // Assign colour based on distance as a percentage of the maximum
  uchar * output_row = nullptr;
  for (int row_index = 0; row_index < dist_matrix.size(); row_index++)
  {
    output_row = heat_map->ptr <uchar> (row_index);
    for (int col_index = 0; col_index < dist_matrix.size(); col_index++)
    {
      output_row[col_index] =
        ((dist_matrix[row_index][col_index] - min_distance)
         / (max_distance - min_distance)) * 255;
    } // for
  } // for
  return heat_map;
} // function heat_map

// Given a probability matrix, normalises it so the probabilities on each
// row add up to 1
void normalise_probabilities(std::vector <std::vector <double>> & matrix)
{
  std::cout << "Normalising probabilities..." << std::endl; // output for debugging

  double row_total = 0;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    // calculate row total
    row_total = 0;
    for (int col_index = 0; col_index < matrix.size(); col_index++)
    {
      row_total += matrix[row_index][col_index];
    } // for

    // normalise probabilities
    for (int col_index = 0; col_index < matrix.size(); col_index++)
    {
      matrix[row_index][col_index] *=
        matrix[row_index][col_index] / row_total;
    } // for
  } // for
} // function normalise_probabilities

// Given a matrix, calculates and returns  a threshold for it based on the
// total distance between adjacent frames.
double calculate_threshold(std::vector <std::vector <double>> & matrix)
{
  double total = 0;
  for (int index = 0; index < matrix.size() - 1; index++)
  {
    total += matrix[index][index + 1];
  } // for
  return (1.0 / (matrix.size() - 1)) * total;
} // function calculate_threshold

// Outputs a representation of the matrix to standard output.
template <class T>
void print_matrix(std::vector <std::vector <T>> & matrix)
{
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix[row_index].size();
         col_index++)
    {
      std::cout << "| " << matrix[row_index][col_index] << ", ";
      std::cout << std::endl;
    } // for
  } // for
} // function print_matrix

Video * load_video(std::string file_name)
{
  cv::VideoCapture capture(file_name);
  if (!capture.isOpened())
  {
    return nullptr;
  } else if (stabilise && !stabilise_video(capture))
  {
    std::cout << "A problem occured while stabilising the video." << std::endl;
    return nullptr;
  } // else if

  std::cout << "Inputting frames from video..." << std::endl; // output for debug

  Video * video = new Video;
  video->set_fps(capture.get(CV_CAP_PROP_FPS));
  video->set_frame_height(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  video->set_frame_width(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int number_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
  for (int frame_num = 0; frame_num < number_frames; frame_num++)
  {
    capture.set(CV_CAP_PROP_POS_FRAMES, frame_num);
    cv::Mat frame;
    if (!capture.read(frame))
    {
      break;
    } // if
    video->get_frames().push_back(frame);
  } // for

  capture.release();
  return video;
} // function load_video

void write_video(Video & video, std::string file_name)
{
  std::cout << "Saving output video..." << std::endl; // output for debugging

  std::vector <cv::Mat> ::iterator it;
  cv::VideoWriter output(file_name, CV_FOURCC('M', 'J', 'P', 'G'),
                         video.get_fps(), cv::Size(video.get_frame_width(),
                                                   video.get_frame_height()));
  for (it = video.get_frames().begin();
       it != video.get_frames().end(); it++)
  {
    output.write(*it);
  } // for
  output.release();
} // function write_video

// Callback function used to respond to events
void on_event(int event, int x, int y,int flags, void * userdata)
{
  int component_length = 1;
  while (component_length * input_video->get_frames().size()
         < min_matrix_display_size)
  {
    component_length++;
  } // while

  if (event == cv::EVENT_LBUTTONDOWN
      && x < (input_video->get_frames().size() - tap_filter) * component_length
      && y + 1 < (input_video->get_frames().size() - tap_filter) * component_length)
  {

    cv::namedWindow("Source", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Source", 0, 0);
    cv::imshow("Source",
               input_video->get_frames()[y / component_length + tap_filter]);

    cv::namedWindow("Destination", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Destination", input_video->get_frame_width() + 3, 0);
    cv::imshow("Destination",
               input_video->get_frames()[x / component_length + tap_filter]);
  } // if
} // function on_mouse_click

// Equalises the brightness of the given video based on a reference image.
void equalise_video_brightness(Video & video, cv::Mat & reference)
{
  std::cout << "Equalising video brightness..." << std::endl; // output for debugging

  int refrence_average = std::floor(average_luminance(reference));
  double scale = 0;
  cv::Mat ycrcb_frame;
  uchar * row = nullptr;
  for (int frame_index = 0; frame_index < video.get_frames().size();
       frame_index++)
  {
    scale = (double) refrence_average
      / average_luminance(video.get_frames()[frame_index]);
    cv::cvtColor(video.get_frames()[frame_index], ycrcb_frame, CV_BGR2YCrCb);

    for (int row_index = 0; row_index < video.get_frame_height(); row_index++)
    {
      row = ycrcb_frame.ptr <uchar> (row_index);
      for (int pix_index = 0;
           pix_index < video.get_frame_width() * ycrcb_frame.channels();
           pix_index += 3)
      {
        row[pix_index] = std::floor(row[pix_index] * scale);
      } // for
    } // for

    cv::cvtColor(ycrcb_frame, video.get_frames()[frame_index], CV_YCrCb2BGR);
  } // for
} // function equalise_video_brightness

// Calculates and returns the average luminance (intensity) of a given image
double average_luminance(cv::Mat & image)
{
  cv::Mat ycrcb_image;
  cv::cvtColor(image, ycrcb_image, CV_BGR2YCrCb);

  uchar * row = nullptr;
  int total = 0;
  for (int row_index = 0; row_index < ycrcb_image.rows; row_index++)
  {
    row = image.ptr <uchar> (row_index);
    for (int pix_index = 0;
         pix_index < ycrcb_image.rows * ycrcb_image.channels();
         pix_index += 3)
    {
      total += row[pix_index];
    } // for
  } // for

  return total / (ycrcb_image.rows * ycrcb_image.cols);
} // function average_luminance

// Finds a portion of the given video that doesn't change much and returns
// a frame from it so it can be used as a reference for brightness
// equalisation.
cv::Mat find_reference_image(Video & video, int num_portions)
{
  std::cout << "Finding reference region for brightness equalisation..."
            << std::endl; // output for debugging

  num_portions = num_portions % 2 == 0 ? num_portions : num_portions + 1;

  std::vector <std::vector <cv::Mat>> portions
    (num_portions, std::vector <cv::Mat> (video.get_frames().size()));
  std::vector <std::vector <double>> lum_averages
    (num_portions, std::vector <double> (video.get_frames().size()));
  std::vector <double> portion_means (num_portions, 0);

  int portion_cols = std::floor(video.get_frame_width() / (num_portions / 2));
  int portion_rows = std::floor(video.get_frame_height() / 2);
  int total = 0;
  double mean = 0;
  double deviation = 0;
  double min_deviation = 100000;
  int min_deviation_index = -1;
  uchar * row = nullptr;
  int row_range[2];
  int col_range[2];
  for (int por_index = 0; por_index < num_portions; por_index++)
  {
    total = 0;

    if (por_index == num_portions / 2)
    {
      // new row
      row_range[0] = portion_rows;
      row_range[1] = portion_rows * 2 - 1;
      col_range[0] = 0;
      col_range[1] = portion_cols - 1;
    } else if (por_index == 0)
    {
      // initialise ranges
      row_range[0] = 0;
      row_range[1] = portion_rows - 1;
      col_range[0] = 0;
      col_range[1] = portion_cols - 1;
    } else
    {
      // change column range
      col_range[0] = col_range[1] + 1;
      col_range[1] = col_range[0] + portion_cols - 1;
    } // else

    // Split each frame into its portions
    for (int frame_index = 0; frame_index < video.get_frames().size();
         frame_index++)
    {
      portions[por_index][frame_index] =
        video.get_frames()[frame_index](cv::Range(row_range[0], row_range[1]),
                                        cv::Range(col_range[0], col_range[1]));
      lum_averages[por_index][frame_index] =
        average_luminance(portions[por_index][frame_index]);
      total += lum_averages[por_index][frame_index];
    } // for

    // Calculate standard deviation of the mean average luminance
    portion_means[por_index] = total / video.get_frames().size();
    deviation = std::abs(standard_deviation(lum_averages[por_index],
                                            portion_means[por_index]));

    if (deviation < min_deviation)
    {
      min_deviation = deviation;
      min_deviation_index = por_index;
    } // if
  } // for

  // Return a frame from portion of the video with the least deviation
  int mean_frame_index = 0; // index of the frame closest to the average
  for (int frame_index = 1; frame_index < video.get_frames().size();
       frame_index++)
  {
    if (std::abs(portion_means[min_deviation_index]
                 - lum_averages[min_deviation_index][frame_index])
        < std::abs(portion_means[min_deviation_index]
                   - lum_averages[min_deviation_index][mean_frame_index]))
    {
      mean_frame_index = frame_index;
    } // if
  } // for

  return portions[min_deviation_index][mean_frame_index];
} // function find_reference_image

// Calculates and returns the standard deviation of a set of values,
// given their mean
double standard_deviation(std::vector <double> values, double mean)
{
  double total = 0;
  for (int index = 0; index < values.size(); index++)
  {
    total += std::pow(values[index] - mean, 2);
  } // for
  return std::sqrt(total / (values.size() - 1));
} // function standard_deviaion

// Finds and returns the maximum component value of the given matrix
template <class T>
T maximum_component_value(std::vector <std::vector <T>> & matrix)
{
  T maximum = 0;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix[0].size(); col_index++)
    {
      if (matrix[row_index][col_index] > maximum)
      {
        maximum = matrix[row_index][col_index];
      } // if
    } // for
  } // for
  return maximum;
} // function maximum_component_value

// Finds and returns the minimum component value of the given matrix
template <class T>
T minimum_component_value(std::vector <std::vector <T>> & matrix,
                          bool non_zero)
{
  T minimum = 10000000;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix[0].size(); col_index++)
    {
      if ((!non_zero || matrix[row_index][col_index] > 0)
          && matrix[row_index][col_index] < minimum)
      {
        minimum = matrix[row_index][col_index];
      } // if
    } // for
  } // for
  return minimum;
} // function minimum_component_value

// Filters the given distance matrix with a diagonal kernel with weights and
// returns a filtered distance matrix
std::vector <std::vector <double>> * preserve_dynamics(
  std::vector <std::vector <double>> & matrix, std::vector <double> & weights)
{
  std::cout << "Filtering distance matrix to preserve dynamics..."
            << std::endl; // output for debugging

  int m = weights.size() / 2;
  std::vector <std::vector <double>> * filtered_matrix =
    create_square_matrix <double> (matrix.size() - (m + 1), 255);

  double total = 0;
  int fil_row_index = 0;
  int fil_col_index = 0;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix.size(); col_index++)
    {
      // set correct index values for filtered matrix ****
      std::cout << "fri = " << fil_row_index << "**  " << filtered_matrix->size() << std::endl; // debugging
      if (fil_row_index >= filtered_matrix->size())
      {
        row_index = matrix.size();
        std::cout << "exiting" << std::endl; // debugging
        break;
      } else if (row_index >= m && col_index >= m
                 && fil_col_index < filtered_matrix->size())
      {
        fil_col_index++;
      } else if (row_index >= m && col_index >= m
                 && fil_row_index < filtered_matrix->size()
                 && fil_col_index >= filtered_matrix->size()) // next row
      {
        fil_col_index = 0;
        fil_row_index++;
      } // else if

      total = 0;
      for (int filter_index = -m; filter_index < m; filter_index++)
      {
        if (row_index + filter_index >= 0
            && row_index + filter_index < matrix.size()
            && col_index + filter_index >= 0
            && col_index + filter_index < matrix.size())
        {
          total += weights[filter_index + m]
            * matrix[row_index + filter_index][col_index + filter_index];
        } // if
      } // for
      std::cout << fil_row_index << ",  " << fil_col_index << std::endl; // debugging
      (*filtered_matrix)[fil_row_index][fil_col_index] = total;
    } // for
  } // for
  exit(0); // debugging
  return filtered_matrix;
} // function preserve_dynamics

std::vector <std::vector <double>> * preserve_dynamics2(
  std::vector <std::vector <double>> & matrix, std::vector <double> & weights)
{
  std::cout << "Filtering distance matrix to preserve dynamics..."
            << std::endl; // output for debugging

  int m = weights.size() / 2;
  std::vector <std::vector <double>> * filtered_matrix =
    create_square_matrix <double> (matrix.size() - (m + 1), 255);

  double total = 0;
  for (int row_index = m; row_index < m + filtered_matrix->size(); row_index++)
  {
    for (int col_index = m; col_index < m + filtered_matrix->size();
         col_index++)
    {
      total = 0;
      for (int filter_index = -m; filter_index < m; filter_index++)
      {
        if (row_index + filter_index >= 0
            && row_index + filter_index < matrix.size()
            && col_index + filter_index >= 0
            && col_index + filter_index < matrix.size())
        {
          total += weights[filter_index + m]
            * matrix[row_index + filter_index][col_index + filter_index];
        } // if
      } // for
      (*filtered_matrix)[row_index - m][col_index - m] = total;
    } // for
  } // for
  return filtered_matrix;
} // function preserve_dynamics2

// Returns a vector consisting of binomial weights to be used as a kernel
std::vector <double> * filter_weights(int num_weights)
{
  std::vector <double> * weights = new std::vector <double> (num_weights);
  for (int index = 0; index < num_weights; index++)
  {
    (*weights)[index] = (double) factorial(num_weights)
      / (factorial(index) * factorial(num_weights - index));
  } // for
  return weights;
} // function filter_weights

// Utility function for calculating the factorial of a given integer
unsigned int factorial(unsigned int n)
{
  return n == 0 ? 1 : n * factorial(n - 1);
} // function factorial

// Calculates the anticipated future cost of a transition from each frame to
// each other frame
std::vector <std::vector <double>> * anticipate_future(
  std::vector <std::vector <double>> & dist_matrix)
{
  std::cout << "Anticipating future costs of transitions..." << std::endl; // output for debugging

  std::vector <std::vector <double>> * cost_matrix =
    new std::vector <std::vector <double>> (
      dist_matrix.size(), std::vector <double> (dist_matrix.size()));

  // Initialise anticipated cost matrix
  for (int row_index = 0; row_index < dist_matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < dist_matrix.size(); col_index++)
    {
      (*cost_matrix)[row_index][col_index] =
        std::pow(dist_matrix[row_index][col_index], multiple_trans_tradeoff);
    } // for
  } // for


  // Repeatedly iterate from the last row to the first and compute anticipated
  // costs until the matrix stabilises
  int total_change = 0;
  double new_value = 0;
  do
  {
    total_change = 0;
    for (int row_index = dist_matrix.size() - 1; row_index >= 0; row_index--)
    {
      for (int col_index = 0; col_index < dist_matrix.size(); col_index++)
      {
        new_value = std::pow(dist_matrix[row_index][col_index],
                             multiple_trans_tradeoff);
          + relative_weight_future_trans
          * minimum_transition(*cost_matrix, col_index);
        total_change +=
          std::abs((*cost_matrix)[row_index][col_index] - new_value);
        (*cost_matrix)[row_index][col_index] = new_value;
      } // for
    } // for
  } while ((double) total_change / dist_matrix.size() > 0.00000000000001);
  return cost_matrix;
} // function anticipate_future

// Finds and returns the minimal cost/distance of a transition from a given
// a given frame to any other frame
double minimum_transition(std::vector <std::vector <double>> & dist_matrix,
                          unsigned int frame_number)
{
  double minimum = dist_matrix[frame_number][0];
  for (int col_index = 1; col_index < dist_matrix[frame_number].size();
       col_index++)
  {
    if (dist_matrix[frame_number][col_index] < minimum)
    {
      minimum = dist_matrix[frame_number][col_index];
    } // if
  } // for
  return minimum;
} // function minimum_transition

// Parses and returns a double from a given string
double s_to_doub(std::string string)
{
  std::istringstream stream(string);
  double value;
  if (!(stream >> value))
  {
    return 0;
  } // if
  return value;
} // function s_to_double

// Displays a list of flags than can be used as command line arguments
void display_help()
{
  std::cout << "Correct usage: ./vtg input_video [flags]"
            << std::endl << std::endl
            << "Input parameters from file: -i" << std::endl
            << "Use luminance as measure of similarity: -l" << std::endl
            << "Equalise video brightness: -b" << std::endl
            << "Stabilise video: -s" << std::endl
            << "Set matrix display size: -m min_width" << std::endl
            << "Set filter tap size: -f num_tap" << std::endl
            << "Anticipate future costs of transitions: -a" << std::endl
            << "Set tradeoff between making single vs. multiple transitions: "
            << "-t value" << std::endl
            << "Set relative weight of future transitions: -w value" << std::endl
            << "Prune transitions: -p" << std::endl
            << "Set number of frames in crossfade: -c value" << std::endl
            << "Output file name: -o file_name.avi"
            << std::endl;
} // function display_help

// Gets rid of all transitions but those that are local maxima
void select_only_local_maxima(
  std::vector <std::vector <double>> & dist_matrix)
{
  std::cout << "Pruning the set of possible transitions..." << std::endl;

  int radius = 5; // filter or radius size
  int local_min[2] = {-1, -1};
  double max_allowed_distance = average_distance(dist_matrix) * 0.7;
  double max_distance = maximum_component_value <double> (dist_matrix);

  // Zero the diagonal
  for (int index = 0; index < dist_matrix.size(); index++)
  {
    dist_matrix[index][index] = max_distance;
  } // for

  for (int row_index = 0; row_index < dist_matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < dist_matrix.size(); col_index++)
    {
      if (dist_matrix[row_index][col_index] == max_distance)
      {
        continue; // skip transitions that are too unlikely
      } // if

      local_min[0] = row_index;
      local_min[1] = col_index;

      for (int region_row = 0; region_row < radius * 2 + 1; region_row++)
      {
        for (int region_col = 0; region_col < radius * 2 + 1; region_col++)
        {
          if (row_index - radius + region_row >= 0
              && row_index - radius + region_row < dist_matrix.size()
              && col_index - radius + region_col >= 0
              && col_index - radius + region_col < dist_matrix.size()
              && local_min[0] >= 0 && local_min[0] < dist_matrix.size()
              && local_min[1] >= 0 && local_min[1] < dist_matrix.size()
              && dist_matrix[row_index - radius + region_row]
                            [col_index - radius + region_col]
              < dist_matrix[local_min[0]][local_min[1]])
          {
            local_min[0] = row_index - radius + region_row;
            local_min[1] = col_index - radius + region_col;
          } // if
        } // for
      } // for

      // set all non-maximal frame distances in the region to a high value
      for (int region_row = 0; region_row < radius * 2 + 1; region_row++)
      {
        for (int region_col = 0; region_col < radius * 2 + 1; region_col++)
        {
          if (row_index - radius + region_row >= 0
              && row_index - radius + region_row < dist_matrix.size()
              && col_index - radius + region_col >= 0
              && col_index - radius + region_col < dist_matrix.size()
              && row_index - radius + region_row != local_min[0]
              && col_index - radius + region_col != local_min[1])
          {
            dist_matrix[row_index - radius + region_row]
                       [col_index - radius + region_col] = max_distance;
          } // if
        } // for
      } // for
    } // for
  } // for
} // function select_only_local_maxima

void select_only_local_maxima2(
  std::vector <std::vector <double>> & dist_matrix)
{
  std::cout << "Pruning the set of possible transitions..." << std::endl;

  double max_distance = maximum_component_value <double> (dist_matrix);
  int square_size = dist_matrix.size() / 10;
  int sqr_start_row = 0, sqr_start_col = 0;
  double sqr_min_row = 0, sqr_min_col = 0;
  int end_row = 0, end_col = 0;

  // Set the diagonal to the maximal distance so it is not considered
  for (int index = 0; index < dist_matrix.size(); index++)
  {
    dist_matrix[index][index] = max_distance;
  } // for

  while (true)
  {
    sqr_min_row = sqr_start_row;
    sqr_min_col = sqr_start_col;
    end_row = sqr_start_row + square_size + dist_matrix.size() % 10;
    end_col = sqr_start_col + square_size + dist_matrix.size() % 10;

    // Find minimum distance in the square region
    for (int sqr_row = sqr_start_row;
         sqr_row < end_row && sqr_row < dist_matrix.size(); sqr_row++)
    {
      for (int sqr_col = sqr_start_col;
           sqr_col < end_col && sqr_col < dist_matrix.size(); sqr_col++)
      {
        if (dist_matrix[sqr_row][sqr_col]
            < dist_matrix[sqr_min_row][sqr_min_col])
        {
          sqr_min_row = sqr_row;
          sqr_min_col = sqr_col;
        } // if
      } // for
    } // for

    // Set all non-minimal distances to the maximum distance
    for (int sqr_row = sqr_start_row;
         sqr_row < end_row && sqr_row < dist_matrix.size(); sqr_row++)
    {
      for (int sqr_col = sqr_start_col;
           sqr_col < end_col && sqr_col < dist_matrix.size(); sqr_col++)
      {
        if (sqr_row != sqr_min_row || sqr_col != sqr_min_col)
        {
          dist_matrix[sqr_row][sqr_col] = max_distance;
        } // of
      } // for
    } // for

    if (end_col < dist_matrix.size())
    {
      sqr_start_col = end_col;
    } else if (end_row < dist_matrix.size())
    {
      sqr_start_row = end_row;
      sqr_start_col = 0;
    } else
    {
      break;
    } // else
  } // while

  // Remove (by maximising) the half of the matrix above the identical
  for (int row_index = 0; row_index < dist_matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < dist_matrix.size(); col_index++)
    {
      if (col_index <= row_index - 5)
      {
        dist_matrix[row_index][col_index] = max_distance;
      } // if
    } // for
  } // for
} // function select_only_local_maxima2

template <class T>
T matrix_total(std::vector <std::vector <T>> & matrix)
{
  T total = 0;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix[0].size(); col_index++)
    {
      total += matrix[row_index][col_index];
    } // for
  } // for
  return total;
} // function matrix_total

// Sets all values below a threshold or above a given threshold to a given
// new value
template <class T>
void threshold_matrix(std::vector <std::vector <double>> & matrix, bool below,
                      T threshold, T new_value)
{
  bool comparison = false;
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix[0].size(); col_index++)
    {
      comparison = below ? matrix[row_index][col_index] < threshold
        : matrix[row_index][col_index] > threshold;
      if (comparison)
      {
        matrix[row_index][col_index] = new_value;
      } // if
    } // for
  } // for
} // function threshold_matrix

// Calculate the average cost for each transition and returns a list of those
// with the lowest average cost
std::list <struct Transition> * lowest_average_cost_transitions(
  std::vector <std::vector <double>> & dist_matrix, int max_remaining)
{
  std::cout << "Sorting transitions based on average cost..." << std::endl;

  double max_distance = maximum_component_value <double> (dist_matrix);
  std::list <struct Transition> transitions;
  std::list <struct Transition> * best_transitions =
    new std::list <struct Transition> ();
  struct Transition * current_transition = nullptr;

  // Calculates average cost for all remaining transitions
  for (int row_index = 0; row_index < dist_matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < dist_matrix.size(); col_index++)
    {
      if (dist_matrix[row_index][col_index] >= max_distance
          || dist_matrix[row_index][col_index] <= 0)
      {
        continue;
      } // if
      current_transition = new struct Transition();
      if (row_index > col_index) // only backwards transitions are allowed
      {
        current_transition->source = row_index;
        current_transition->destination = col_index;
      } else
      {
        current_transition->source = col_index;
        current_transition->destination = row_index;
      } // else
      current_transition->average_cost = dist_matrix[row_index][col_index];
      transitions.push_back(*current_transition);
    } // for
  } // for

  // Sort transitions based on average costs
  transitions.sort(compare_transitions);

  // Return small number of best transitions
  while (max_remaining > transitions.size())
  {
    max_remaining--;
  } // while
  best_transitions->splice(best_transitions->end(), transitions,
                           transitions.begin(), std::next(transitions.begin(),
                                                          max_remaining));
  return best_transitions;
} // function lowest_average_cost_transitions

// Given two transitions, returns true if the first is less than or equal to the
// second, or false otherwise.
bool compare_transitions(struct Transition & first,
                         struct Transition & second)
{
  return first.average_cost <= second.average_cost;
} // function compare_transitions

// Utility function which creates and returns a vector containing all of the
// elements in a list.
template <class T>
std::vector <T> * vector_from_list(std::list <T> & list)
{
  std::vector <T> * vector = new std::vector <T> (list.size());
  int index = 0;
  for (typename std::list <T> ::const_iterator it = list.begin();
       it != list.end(); it++)
  {
    (*vector)[index] = *it;
    index++;
  } // for
  return vector;
} // function vector_from_list

// Creates and returns a dynamic programming table used to find optimal loops
std::vector <std::vector <struct CompoundLoop>> * dp_table(
  std::list <struct Transition> & transition_list, int max_loop_length)
{
  std::cout << "Selecting a set of transitions to use..." << std::endl;

  // Find the maximum primitive loop length
  int max_primitive_loop_length = max_loop_length;
  for (std::list <struct Transition> ::const_iterator it =
        transition_list.begin(); it != transition_list.end(); it++)
  {
    if (transition_length(*it) > max_primitive_loop_length)
    {
      max_primitive_loop_length = transition_length(*it);
    } // if
  } // for
  max_primitive_loop_length *= 3;

  // Store transitions in a vector and initialise the DP table
  std::vector <struct Transition> * transitions =
    vector_from_list <struct Transition> (transition_list);
  std::vector <std::vector <struct CompoundLoop>> * table =
    new std::vector <std::vector <struct CompoundLoop>> (
      max_primitive_loop_length + 1, // row 0 is left empty
      std::vector <struct CompoundLoop> (transitions->size()));

  // Add primitive loops to their columns
  int length = 0;
  for (int col = 0; col < transitions->size(); col++)
  {
    length = transition_length((*transitions)[col]);
    (*table)[length][col].transitions.push_back((*transitions)[col]);
    (*table)[length][col].total_cost = (*transitions)[col].average_cost;
  } // for

  // Main DP loop
  struct CompoundLoop * lowest_cost_loop = nullptr;
  struct CompoundLoop * loop_considered = nullptr;
  int compound_length = 0;
  for (int row = 1; row <= max_primitive_loop_length; row++)
  {
    for (int col = 0; col < transitions->size(); col++)
    {
      // Skip cells of length less than the column's primitive loop length
      if (row < transition_length((*transitions)[col]))
      {
        continue;
      } // if

      lowest_cost_loop = nullptr;

      // Check compound loops of shorter length in same column
      for (int offset = 1; offset < row; offset++)
      {
        if ((*table)[row - offset][col].total_cost == 0)
        {
          continue;
        } // if

        if (row - offset > 0
            && row >= transition_length((*transitions)[col]))
        {
          // Try to add compound loops from colums whose primitive loops have
          // ranges which overlap that of the column being considered
          for (int cmp_col = 0; cmp_col < transitions->size(); cmp_col++)
          {
            if (transition_length((*transitions)[cmp_col])
                + compound_loop_length((*table)[row - offset][col]) != row
                || !transition_ranges_overlap(
                      (*table)[row - offset][col].transitions,
                      (*transitions)[cmp_col]))
            {
              continue;
            } // if

            for (int cmp_row = 1; cmp_row <= max_primitive_loop_length;
                 cmp_row++)
            {
              if ((*table)[cmp_row][cmp_col].total_cost == 0)
              {
                continue;
              } else
              {
                compound_length =
                  compound_loop_length((*table)[row - offset][col])
                  + compound_loop_length((*table)[cmp_row][cmp_col]);
                if (compound_length > row)
                {
                  break;
                } else if (compound_length != row)
                {
                  continue;
                } // else if
              } // else if

              // Check if loop under consideration is better than current best
              loop_considered =
                merge_compound_loops((*table)[row - offset][col],
                                     (*table)[cmp_row][cmp_col]);
              if (lowest_cost_loop == nullptr
                  || loop_considered->total_cost
                  < lowest_cost_loop->total_cost)
              {
                lowest_cost_loop = loop_considered;
              } // if
            } // for
          } // for
        } // if
      } // for

      if (lowest_cost_loop != nullptr)
      {
        (*table)[row][col] = *lowest_cost_loop;
      } // if
    } // for
  } // for
  return table;
} // function dp_table

// Returns true if the ranges the given list of primitive loops overlap the given
// primitive loops or false otherwise
bool transition_ranges_overlap(std::list <struct Transition> transitions,
                               struct Transition primitive)
{
  if (transitions.empty())
  {
    return false;
  } // if

  int min = transitions.front().source;
  int max = transitions.front().destination;
  std::list <struct Transition> ::const_iterator it;
  for (it = transitions.begin(); it != transitions.end(); it++)
  {
    if ((*it).source < min)
    {
      min = (*it).source;
    } else if ((*it).destination > max)
    {
      max = (*it).destination;
    } // else if
  } // for

  return primitive.source >= min && primitive.source <= max
    || primitive.destination >= min && primitive.destination <= max;
} // function transition_ranges_overlap

// Returns the length of a given compound loop
int compound_loop_length(struct CompoundLoop & loop)
{
  if (loop.transitions.size() < 1)
  {
    return 0;
  } // if

  int total = 0;
  std::list <struct Transition> ::const_iterator it;
  for (it = loop.transitions.begin(); it != loop.transitions.end(); it++)
  {
    total += transition_length(*it);
  } // for

  return total;
} // function compound_loop_length;

// Given two compound loops, returns a compound loop which is a combination
// of the first and second
struct CompoundLoop * merge_compound_loops(struct CompoundLoop & first,
                                           struct CompoundLoop & second)
{
  struct CompoundLoop * loop = new struct CompoundLoop;
  std::list <struct Transition> ::const_iterator it;

  loop->transitions = first.transitions;
  loop->total_cost = first.total_cost;
  for (it = second.transitions.begin(); it != second.transitions.end(); it++)
  {
    loop->transitions.push_back(*it);
    loop->total_cost += (*it).average_cost;
  } // for
  return loop;
} // function merge_compound_loops

// Outputs the given list of transitions to standard output. Mainly used for
// debugging purposes.
void print_transitions_list(std::list <struct Transition> & transitions)
{
  std::cout << "Transitions List: (" << transitions.size() << ")" <<std::endl;
  std::list <struct Transition> ::const_iterator it;
  int count = 0;
  for (it = transitions.begin(); it != transitions.end(); it++)
  {
    std::cout << count << ") Source: " << (*it).source << std::endl
              << "    Destination: " << (*it).destination << std::endl
              << "    Range: " << transition_length(*it) << std::endl;
    count++;
  } // for
} // function print_transitions_list

// Returns the length of a transition
int transition_length(struct Transition transition)
{
  return std::abs(transition.destination - transition.source);
} // function transition_length

// Given a non-empty compound loop, schedules and updates the list of
// transitions so they form a valid compound loop
void schedule_transitions(struct CompoundLoop & loop)
{
  std::cout << "Scheduling primitive loops to form a valid compound loop..."
            << std::endl; // output for debugging

  std::list <struct Transition> schedule;
  std::list <std::list <struct Transition>> * ranges = nullptr;
  std::list <std::list <struct Transition>> * new_ranges = nullptr;
  std::list <std::list <struct Transition>> ::iterator range_it;
  std::list <struct Transition> ::iterator trans_it;
  struct Transition removed = remove_latest_transition(loop.transitions);
  schedule.push_back(removed);
  ranges = continuous_ranges(loop.transitions);
  for (range_it = ranges->begin(); range_it != ranges->end();)
  {
    for (trans_it = (*range_it).begin(); trans_it != (*range_it).end();)
    {
      if ((*trans_it).source > removed.destination)
      {
        removed = *trans_it;
        schedule.push_back(removed);
        (*range_it).erase(trans_it);
        trans_it = (*range_it).begin();

        // split range into continuous ranges
        new_ranges = continuous_ranges(*range_it);
        ranges->erase(range_it);
        for (range_it = new_ranges->begin(); range_it != new_ranges->end();
             range_it++)
        {
          ranges->push_front(*range_it);
        } // for
        range_it = ranges->begin();
        break;
      } // if
      trans_it++;
    } // for
    range_it++;
  } // for
  loop.transitions = schedule;
} // function schedule_transitions

// Removes and returns the transition in the given list which starts at
// the latest point in the sequence
struct Transition remove_latest_transition(
  std::list <struct Transition> & transitions)
{
  std::list <struct Transition> ::const_iterator latest = transitions.begin();
  std::list <struct Transition> ::const_iterator it;
  for (it = std::next(transitions.begin()); it != transitions.end();
       it++)
  {
    if ((*it).source > (*latest).source)
    {
      latest = it;
    } // if
  } // for
  struct Transition latest_transition = *latest;
  transitions.erase(latest);
  return latest_transition;
} // function remove_latest_transition

// Splits the given list of transitions into several with continuous ranges.
std::list <std::list <struct Transition>> * continuous_ranges(
  std::list <struct Transition> transitions)
{
  std::list <struct Transition> ::iterator trans_it;
  std::list <std::list <struct Transition>> * ranges =
    new std::list <std::list <struct Transition>>();
  bool range_matched = false;
  std::list <std::list <struct Transition>> ::iterator range_it;
  for (trans_it = transitions.begin(); trans_it != transitions.end();
       trans_it++)
  {
    range_matched = false;
    for (range_it = ranges->begin(); range_it != ranges->end();
         range_it++)
    {
      if (transition_ranges_overlap(*range_it, *trans_it))
      {
        (*range_it).push_back(*trans_it);
        break;
      } // if
    } // for

    if (!range_matched)
    {
      std::list <struct Transition> new_range;
      new_range.push_back(*trans_it);
      ranges->push_back(new_range);
    } // if
  } // for

  return ranges;
} // function continuous_ranges

struct PreRenderedSequence * create_loop_frame_sequence(
  struct CompoundLoop & loop, std::string filename)
{
  std::ofstream file (filename);
  std::cout << "Generating frame sequence from the compound loop..."
            << std::endl; // output for debugging

  std::list <struct Transition> ::const_iterator trans_it;
  std::list <cv::Mat> frames;
  std::list <int> transition_points;
  int frame_counter = 0;
  int start_frame = 0;
  int end_frame = 0;
  int frame_index = 0;

  for (trans_it = loop.transitions.begin();
       trans_it != loop.transitions.end(); trans_it++)
  {
    start_frame = trans_it == loop.transitions.begin() ?
      loop.transitions.back().destination + 1
      : (*std::prev(trans_it)).destination + 1;
    end_frame = (*trans_it).source - 1;
    for (frame_index = start_frame; frame_index < end_frame; frame_index++)
    {
      file << frame_index << std::endl;
      frames.push_back(input_video->get_frames()[frame_index]);
      frame_counter++;
    } // for
    transition_points.push_back(frame_counter);
    frames.push_back(input_video->get_frames()[(*trans_it).source]);
    frames.push_back(input_video->get_frames()[(*trans_it).destination]);
    file << (*trans_it).source << " ->  " << (*trans_it).destination << std::endl;
    frame_counter += 2;
  } // for

  file.close();
  struct PreRenderedSequence * sequence = new struct PreRenderedSequence;
  sequence->frames = vector_from_list <cv::Mat> (frames);
  sequence->transitions = transition_points;
  return sequence;
} // function create_loop_frame_sequence

cv::Mat blend_frames(cv::Mat & frame_a, float alpha, cv::Mat & frame_b,
                     float beta)
{
  cv::Mat result = frame_a.clone();
  int rows = frame_a.rows;
  int cols = frame_a.cols;
  int channels = frame_a.channels();
  if (frame_a.isContinuous())
  {
    cols *= rows;
    rows = 1;
  } // if

  uchar * row_a, * row_b, * result_row;
  for (int row_index = 0; row_index < rows; row_index++)
  {
    row_a = frame_a.ptr <uchar> (row_index);
    row_b = frame_b.ptr <uchar> (row_index);
    result_row = result.ptr <uchar> (row_index);
    for (int pix_index = 0; pix_index < cols * channels;
         pix_index += 3)
    {
      result_row[pix_index] = (uchar)((float) row_a[pix_index] * alpha
        + (float) row_b[pix_index] * beta);
      result_row[pix_index + 1] = (uchar)((float) row_a[pix_index + 1] * alpha
        + (float) row_b[pix_index + 1] * beta);
      result_row[pix_index + 2] = (uchar)((float) row_a[pix_index + 2] * alpha
        + (float) row_b[pix_index + 2] * beta);
    } // for
  } // for
  return result;
} // function blend_frames

std::vector <cv::Mat> * blend_frames_at_transitions(
  std::vector <cv::Mat> & frames, std::list <int> & transition_points,
  int frames_per_blend)
{
  std::cout << "Blending frames at transition points..." << std::endl;

  std::vector <cv::Mat> * previous_frames = new std::vector <cv::Mat>();
  *previous_frames = frames;
  std::vector <cv::Mat> * current_frames = new std::vector <cv::Mat>();
  *current_frames = *previous_frames;

  std::vector <float> * weights = nullptr;
  if (frames_per_blend == 4)
  {
    weights = new std::vector <float> (4);
    (*weights)[0] = 0.5;
    (*weights)[1] = 0.3;
    (*weights)[2] = 0.1;
    (*weights)[3] = 0.1;
  } else if (frames_per_blend == 6)
  {
    weights = new std::vector <float> (6);
    (*weights)[0] = 0.4;
    (*weights)[1] = 0.2;
    (*weights)[2] = 0.1;
    (*weights)[3] = 0.1;
    (*weights)[4] = 0.1;
    (*weights)[5] = 0.1;
  } else // 2 frames per blend
  {
    frames_per_blend = 2;
    weights = new std::vector <float> (2);
    (*weights)[0] = 0.6;
    (*weights)[1] = 0.4;
  } // else
  alpha_weights = weights;
  int current_frames_per_blend = frames_per_blend;

  std::list <int> ::const_iterator it;
  for (it = transition_points.begin(); it != transition_points.end(); it++)
  {
    // Blend frames
    uchar * row = nullptr;
    float blue = 0, green = 0, red = 0;
    int max_blue = 0, max_green = 0, max_red = 0;
    for (int centre_frame = *it - current_frames_per_blend / 2 + 1;
         centre_frame < *it + current_frames_per_blend / 2 + 1; centre_frame++)
    {
      for (int row_index = 0; row_index < input_video->get_frame_height();
           row_index++)
      {
        row = (*current_frames)[correct_frame_index(
          frames.size(), centre_frame)].ptr <uchar> (row_index);
        for (int pix_index = 0;
             pix_index < input_video->get_frame_width(); pix_index++)
        {
          blue = 0; green = 0; red = 0;
          max_blue = 0; max_green = 0; max_red = 0;
          for (int frame_index = *it - current_frames_per_blend / 2 + 1;
               frame_index < *it + current_frames_per_blend / 2 + 1;
               frame_index++)
          {
            // Calculate maximum RGB values
            if (((*previous_frames)[correct_frame_index(
                  frames.size(), frame_index)].at <cv::Vec3b> (
                    row_index, pix_index))[0] > max_blue)
            {
              max_blue = ((*previous_frames)[correct_frame_index(
                frames.size(), frame_index)].at <cv::Vec3b> (
                  row_index, pix_index))[0];
            } // if
            if (((*previous_frames)[correct_frame_index(
                  frames.size(), frame_index)].at <cv::Vec3b> (
                    row_index, pix_index))[1] > max_green)
            {
              max_green = ((*previous_frames)[correct_frame_index(
                frames.size(), frame_index)].at <cv::Vec3b> (
                  row_index, pix_index))[1];
            } // if
            if (((*previous_frames)[correct_frame_index(
                  frames.size(), frame_index)].at <cv::Vec3b> (
                    row_index, pix_index))[2] > max_red)
            {
              max_red = ((*previous_frames)[correct_frame_index(
                frames.size(), frame_index)].at <cv::Vec3b> (
                  row_index, pix_index))[2];
            } // if

            // Add contributions of each frame to the blended frame
            blue += (*weights)[std::abs(centre_frame - frame_index)]
              * ((*previous_frames)[correct_frame_index(
                   frames.size(), frame_index)].at <cv::Vec3b> (
                      row_index, pix_index))[0];
            green += (*weights)[std::abs(centre_frame - frame_index)]
              * ((*previous_frames)[correct_frame_index(
                   frames.size(), frame_index)].at <cv::Vec3b> (
                     row_index, pix_index))[1];
            red += (*weights)[std::abs(centre_frame - frame_index)]
              * ((*previous_frames)[correct_frame_index(
                   frames.size(), frame_index)].at <cv::Vec3b> (
                     row_index, pix_index))[2];
          } // for
          row[pix_index * 3] = ((int)blue <= max_blue) ? blue : max_blue;
          row[pix_index * 3 + 1] = ((int)green <= max_green) ? green
            : max_green;
          row[pix_index * 3 + 2] = ((int)red <= max_red) ? red : max_red;
        } // for
      } // for
    } // for

    delete previous_frames;
    previous_frames = current_frames;
    *current_frames = *previous_frames;
  } // for

  return current_frames;
} // function blend_frames_at_transitions

int correct_frame_index(int sequence_length, int frame_index)
{
  if (frame_index > sequence_length - 1)
  {
    return frame_index - sequence_length;
  } else if (frame_index < 0)
  {
    return sequence_length + frame_index;
  } // else if
  return frame_index;
} // function correct_frame_number

void save_parameters(std::string filepath)
{
  std::ofstream file (filepath);
  file << "Brightness equalised = " << equalise_brightness
       << "\nTap filter size = " << tap_filter
       << "\nMultiple transition tradeoff = " << multiple_trans_tradeoff
       << "\nRelative weight of future transitions = "
       << relative_weight_future_trans
       << "\nNumber frames per transition blend = " << num_frames_to_blend;
  if (alpha_weights != nullptr)
  {
    file << "\nBlend weights = [   ";
    for (int index = 0; index < alpha_weights->size(); index++)
    {
      file << (*alpha_weights)[index] << "   ";
    } // for
    file << " ]";
  } // if
  file.close();
} // function save_parameters

void save_transition_pairs(std::vector <cv::Mat> & frames,
                           std::list <int> & transitions, std::string filepath)
{
  int transition_num = 1;
  std::list <int> ::iterator it;
  for (it = transitions.begin(); it != transitions.end(); it++)
  {
    cv::imwrite(filepath + "/" + std::to_string(transition_num) + "_a.jpg",
                frames[*it]);
    cv::imwrite(filepath + "/" + std::to_string(transition_num) + "_b.jpg",
                frames[*it + 1]);
    transition_num++;
  } // for
} // save_transition_pairs

void setup_output_dirs(std::string filepath)
{
  boost::filesystem::create_directory(filepath);
} // function setup_output_dirs

void save_matrix(std::vector <std::vector <double>> & matrix,
                 std::string filepath)
{
  cv::Mat * image = heat_map(matrix);
  image = enlarge_matrix(*image);
  cv::imwrite(filepath, *image);
} // function save_matrix

std::string output_name()
{
  std::string name = "";
  if (sim_measure == rgb)
  {
    name.push_back('r');
  } else
  {
    name.push_back('y');
  } // else

  if (equalise_brightness)
  {
    name.push_back('B');
  } else
  {
    name.push_back('b');
  } // else

  char mtp[7];
  char rwft[7];
  sprintf(mtp, "%.2f", multiple_trans_tradeoff);
  sprintf(rwft, "%.2f", relative_weight_future_trans);
  name += "f" + std::to_string(tap_filter);
  name += "t" + std::string(mtp);
  name += "w" + std::string(rwft);
  name += "c" + std::to_string(num_frames_to_blend);
  return name;
} // function output_name

void save_distance_values(std::vector <std::vector <double>> & matrix,
                          std::string filepath)
{
  std::cout << "Saving distance values to file..." << std::endl;

  std::ofstream file (filepath);
  for (int row_index = 0; row_index < matrix.size(); row_index++)
  {
    for (int col_index = 0; col_index < matrix.size(); col_index++)
    {
      file << std::setprecision(20) << matrix[row_index][col_index] << ",";
    } // for
    file << std::endl;
  } // for
  file.close();
} // function save_distance_values

std::vector <std::vector <double>> * load_distance_matrix(
  std::string filepath)
{
  std::cout << "Loading distance matrix from file..." << std::endl;

  std::vector <std::vector <double>> * matrix = nullptr;
  std::list <std::list <double>> rows;
  std::list <double> current_row;
  std::ifstream file (filepath.c_str());
  char current_char = '\0';
  char prev_char = '\0';
  std::string current_value = "";
  while (file.get(current_char))
  {
    if (current_char == '\n')
    {
      if (prev_char != ',' && current_value != "")
      {
        current_row.push_back(std::stod(current_value));
        current_value = "";
      } // if
      if (!current_row.empty())
      {
        rows.push_back(current_row);
        current_row.clear();
      } // if
    } else if (current_char == ',')
    {
      current_row.push_back(std::stod(current_value));
      current_value = "";
    } else if (isdigit(current_char) || current_char == '.'
               || current_char == 'e' || current_char == '+')
    {
      current_value.push_back(current_char);
    } // else
    prev_char = current_char;
  } // while
  file.close();

  matrix = create_square_matrix <double> (input_video->get_frames().size(), 0);
  std::vector <double> * vector_row = nullptr;
  std::list <std::list <double>> ::const_iterator row_it = rows.begin();
  std::list <double> ::const_iterator val_it;
  for (int row_index = 0;
       row_index < matrix->size() && row_it != rows.end();
       row_index++, row_it++)
  {
    val_it = (*row_it).begin();
    for (int col_index = 0;
         col_index < matrix->size() && val_it != (*row_it).end();
         col_index++, val_it++)
    {
      (*matrix)[row_index][col_index] = *val_it;
    } // for
  } // for
  return matrix;
} // function load_distance_matrix

void load_parameters_from_string(std::string parameters)
{
  std::cout << "Setting parameters for current test..." << std::endl;

  std::string current_value = "";
  char param_type = '\0';
  for (int index = 0; index < parameters.size(); index++)
  {
    if (isalpha(parameters[index]) && parameters[index + 1] == '=')
    {
      param_type = parameters[index];
      index++;
    } else if (parameters[index] == ',' || index == parameters.size() - 1)
    {
      if (parameters[index] != ',' && index == parameters.size() - 1)
      {
        current_value.push_back(parameters[index]);
      } // if
      switch (param_type)
      {
        case 'r':
          sim_measure = std::stoi(current_value) == 0 ? rgb : luminance;
          break;
        case 'b':
          equalise_brightness = std::stoi(current_value);
          break;
        case 'c':
          num_frames_to_blend = std::stoi(current_value);
          break;
        case 't':
          multiple_trans_tradeoff = std::stod(current_value);
          break;
        case 'f':
          tap_filter = std::stoi(current_value);
          break;
        case 'w':
          relative_weight_future_trans = std::stod(current_value);
          break;
        default: break;
      } // switch
      current_value = "";
    } else if (isdigit(parameters[index]) || parameters[index] == '.')
    {
      current_value.push_back(parameters[index]);
    } // else if
  } // for
} // function load_parameters_from_string

std::list <std::string> load_parameters_from_file(std::string filename)
{
  std::ifstream file (filename);
  std::list <std::string> parameter_sets;
  char buffer[200];
  while (file.getline(buffer, 200))
  {
    parameter_sets.push_back(std::string(buffer));
  } // while
  file.close();
  return parameter_sets;
} // function load_parameters_from_file
