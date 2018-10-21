#include "vtg.h"
#include <iostream>
#include <iterator>
#include <cmath>

enum similarity_measure sim_measure = rgb;
Video * input_video = nullptr;
bool equalise_brightness = false;
bool stabilise = false;
int min_matrix_display_size = 500;

int main(int argc, char ** argv)
{
  std::string output_file_name = "output.avi";

  if (!check_flags(argc, argv))
  {
    return 0;
  } // if

  input_video = load_video(argv[1]);
  if (input_video == nullptr)
  {
    std::cout << "Input file provided is not a valid video." << std::endl;
    return 0;
  } // if

  apply_preprocessing(*input_video);
  std::vector <std::vector <double>> * distances =
    distance_matrix(*input_video);
  std::vector <std::vector <double>> * probabilities =
    probability_matrix(*distances, 5 * average_distance(*distances));
  normalise_probabilities(*probabilities);
  display_matrix(*probabilities);

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
  } // if

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
        if (argv[arg_index][char_index] == 'l')
        {
          sim_measure = luminance;
        } else if (argv[arg_index][char_index] == 'b')
        {
          equalise_brightness = true;
        } else if (argv[arg_index][char_index] == 's')
        {
          stabilise = true;
        } else if (argv[arg_index][char_index] == 'm'
                   && argc > arg_index)
        {
          arg_index++;
          min_matrix_display_size = std::stoi(argv[arg_index]);
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

void apply_preprocessing(Video & video)
{
  if (equalise_brightness)
  {
    cv::Mat reference_image = find_reference_image(*input_video, 6);
    equalise_video_brightness(video, reference_image);
  } // if

  if (stabilise)
  {
  } // if
} // function apply_preprocessing

// Takes a similarity measure and a video.
// Returns a pointer to a distance matrix of the frames in the video.
std::vector <std::vector <double>> * distance_matrix(Video & video)
{
  std::vector <std::vector <double>> * matrix =
    create_square_matrix <double> (video.get_frames().size(), 0);

  std::cout << "Analysing frames..." << std::endl;

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
      } // else
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
  std::cout << "Calculating average distances..." << std::endl; // output for debug

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

// Displays the given matrix in a window after thresholding it
void display_matrix(std::vector <std::vector <double>> & matrix)
{
  std::cout << "Displaying matrix..." << std::endl; // output for debugging
  cv::Mat * binary_mat = threshold_matrix(matrix, calculate_threshold(matrix));

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
} // function display_matrix

// Returns a "binary" Mat object with components set to 0 if their
// corresponding components in the original matrix are greater than the given
// threshold, or set to 255 otherwise.
cv::Mat * threshold_matrix(std::vector <std::vector <double>> & matrix,
                           double threshold)
{
  cv::Mat * output = new cv::Mat(matrix.size(), matrix.size(), CV_8U,
                                 cv::Scalar::all(255));
  uchar * output_row;
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
} // function threshold_matrix

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
  } // if

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

void write_video(Video & video, std::string & file_name)
{
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
      && x < input_video->get_frames().size() * component_length
      && y + 1 < input_video->get_frames().size() * component_length)
  {

    cv::namedWindow("FirstFrame", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("FirstFrame", 0, 0);
    cv::imshow("FirstFrame", input_video->get_frames()[y / component_length]);

    cv::namedWindow("SecondFrame", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("SecondFrame", input_video->get_frame_width() + 3, 0);
    cv::imshow("SecondFrame", input_video->get_frames()[x / component_length]);
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
        //std::cout << "old: " << (int)row[pix_index]; // output for debugging
        row[pix_index] = std::floor(row[pix_index] * scale);
        //std::cout << ", new: " << (int)row[pix_index] << std::endl; // output for debugging
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

    std::cout << "Portion " << por_index << ": rows = "
              << row_range[0] << " to " << row_range[1] << ", cols = "
              << col_range[0] << " to " << col_range[1] << ", mean = "
              << portion_means[por_index] << ", deviation = "
              << deviation << std::endl; // output for debugging

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
