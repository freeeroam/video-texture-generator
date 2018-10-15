#include "vtg.h"
#include <iostream>
#include <iterator>
#include <cmath>

enum similarity_measure sim_measure = rgb;
Video * input_video = nullptr;

int main(int argc, char ** argv)
{
  std::string output_file_name = "output.avi";

  if (argc < 2)
  {
    std::cout << "No input file provided" << std::endl;
    std::cout << "Correct usage: ./vtg filepath" << std::endl;
    return 0;
  } else if (argc == 3)
  {
    sim_measure = argv[2] == "rgb" ? rgb : luminance;
  } // else if

  input_video = load_video(argv[1]);
  if (input_video == nullptr)
  {
    std::cout << "Input file provided is not a valid video." << std::endl;
    return 0;
  } // if

  std::vector <std::vector <double>> * distances =
    distance_matrix(*input_video);
  std::vector <std::vector <double>> * probabilities =
    probability_matrix(*distances, 2 * average_distance(*distances));
  normalise_probabilities(*probabilities);
  display_matrix(*probabilities);

  return 0;
} // function main

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
      if (row_index != col_index)
      {
        (*matrix)[row_index][col_index] =
          calculate_distance(video.get_frames()[row_index],
                             video.get_frames()[col_index]);
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
        val_a = 0.299 * row_a[2]
          + 0.587 * row_a[1]
          + 0.114 * row_a[0];
        val_b = 0.299 * row_b[2]
          + 0.587 * row_b[1]
          + 0.114 * row_b[0];
        total += std::pow(val_a - val_b, 2);
      } else
      {
        total += std::pow(row_a[0] - row_b[0], 2)
          + std::pow(row_a[1] - row_b[1], 2)
          + std::pow(row_a[2] - row_b[2], 2);
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
  cv::Mat * image = threshold_matrix(matrix, calculate_threshold(matrix));
  cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Display", on_event, nullptr);
  cv::imshow("Display", *image);
  cv::waitKey(60000);
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
  std::cout << "Normalizing probabilities..." << std::endl; // output for debugging

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

void on_event(int event, int x, int y,int flags, void * userdata)
{
  if (event == cv::EVENT_LBUTTONDOWN
      && x < input_video->get_frames().size()
      && y < input_video->get_frames().size())
  {
    cv::namedWindow("FirstFrame", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("FirstFrame", 0, 0);
    cv::imshow("FirstFrame", input_video->get_frames()[y + 1]);

    cv::namedWindow("SecondFrame", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("SecondFrame", input_video->get_frame_width() + 3, 0);
    cv::imshow("SecondFrame", input_video->get_frames()[x]);

    cv::waitKey(1000);
  } // if
} // function on_mouse_click
