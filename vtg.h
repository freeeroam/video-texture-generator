#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

class Video
{
  private:
    std::vector <cv::Mat> frames;
    int frame_height;
    int frame_width;
    double fps;
  public:
    Video();
    Video(std::vector <cv::Mat> * frames, double fps, int frame_width,
          int frame_height);
    std::vector <cv::Mat> & get_frames();
    int get_frame_height() const;
    int get_frame_width() const;
    double get_fps() const;
    void set_frames(std::vector <cv::Mat> & frames);
    void set_frame_height(int frame_height);
    void set_frame_width(int frame_width);
    void set_fps(double fps);
    bool operator==(Video & other_video);
}; // class Video

// Different measures of similarity
enum similarity_measure : unsigned int
{
  rgb = 0,
  luminance = 1
}; // enum similarity_measures

// Global function declarations
bool check_flags(int argc, char ** argv);
void apply_preprocessing(Video & video);
std::vector <std::vector <double>> * distance_matrix(Video & video);
template <class T>
std::vector <std::vector <T>> * create_square_matrix(unsigned int size,
                                                     T initial_value);
double calculate_distance(cv::Mat & frame_a, cv::Mat & frame_b);
std::vector <std::vector <double>> * probability_matrix(
  std::vector <std::vector <double>> & distance_matrix, double sigma);
double average_distance(std::vector <std::vector <double>> & matrix);
void display_binary_matrix(std::vector <std::vector <double>> & matrix);
cv::Mat * enlarge_matrix(cv::Mat & matrix);
void display_distance_matrix(std::vector <std::vector <double>> & dist_matrix);
void display_transition_matrix(
  std::vector <std::vector <double>> & prob_matrix);
cv::Mat * threshold_matrix(std::vector <std::vector <double>> & matrix,
                         double threshold);
cv::Mat * heat_map(std::vector <std::vector <double>> & dist_matrix);
double calculate_threshold(std::vector <std::vector <double>> & matrix);
void normalise_probabilities(std::vector <std::vector <double>> & matrix);
template <class T>
void print_matrix(std::vector <std::vector <T>> & matrix);
Video * load_video(std::string file_name);
void write_video(Video & video, std::string & file_name);
void on_event(int event, int x, int y,int flags, void * userdata);
void equalise_video_brightness(Video & video, cv::Mat & reference);
double average_luminance(cv::Mat & image);
cv::Mat find_reference_image(Video & video, int num_portions);
double standard_deviation(std::vector <double> values, double mean);
template <class T>
T maximum_component_value(std::vector <std::vector <T>> & matrix);
