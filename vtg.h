#include <string>
#include <vector>
#include <list>
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

struct Transition
{
  int source;
  int destination;
  double average_cost;
}; // struct Transition

struct CompoundLoop
{
  std::list <struct Transition> transitions;
  double total_cost;

  CompoundLoop() : total_cost(0) {}
}; // struct CompoundLoop

struct PreRenderedSequence
{
  std::vector <cv::Mat> * frames;
  std::list <int> transitions;
}; // struct PreRenderedSequence

struct LoopFrame
{
  unsigned int frame_num; // frame no. in the original input video
  bool transition_point;

  LoopFrame() : frame_num(0), transition_point(false) {}
}; // struct LoopFrame

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
cv::Mat * create_binary_matrix(std::vector <std::vector <double>> & matrix,
                               double threshold);
cv::Mat * heat_map(std::vector <std::vector <double>> & dist_matrix);
double calculate_threshold(std::vector <std::vector <double>> & matrix);
void normalise_probabilities(std::vector <std::vector <double>> & matrix);
template <class T>
void print_matrix(std::vector <std::vector <T>> & matrix);
Video * load_video(std::string file_name);
void write_video(Video & video, std::string file_name);
void on_event(int event, int x, int y,int flags, void * userdata);
void equalise_video_brightness(Video & video, cv::Mat & reference);
double average_luminance(cv::Mat & image);
cv::Mat find_reference_image(Video & video, int num_portions);
double standard_deviation(std::vector <double> values, double mean);
template <class T>
T maximum_component_value(std::vector <std::vector <T>> & matrix);
template <class T>
T minimum_component_value(std::vector <std::vector <T>> & matrix,
                          bool non_zero);
bool stabilise_video(cv::VideoCapture & cap);
std::vector <std::vector <double>> * preserve_dynamics(
  std::vector <std::vector <double>> & matrix, std::vector <double> & weights);
std::vector <std::vector <double>> * preserve_dynamics2(
  std::vector <std::vector <double>> & matrix, std::vector <double> & weights);
std::vector <double> * filter_weights(int num_weights);
unsigned int factorial(unsigned int n);
std::vector <std::vector <double>> * anticipate_future(
  std::vector <std::vector <double>> & dist_matrix);
double minimum_transition(std::vector <std::vector <double>> & dist_matrix,
                          unsigned int frame_number);
double s_to_doub(std::string string);
void display_help();
void select_only_local_maxima(
  std::vector <std::vector <double>> & dist_matrix);
void select_only_local_maxima2(
  std::vector <std::vector <double>> & dist_matrix);
template <class T>
T matrix_total(std::vector <std::vector <T>> & matrix);
template <class T>
void threshold_matrix(std::vector <std::vector <double>> & matrix, bool below,
                      T threshold, T new_value);
std::list <struct Transition> * lowest_average_cost_transitions(
  std::vector <std::vector <double>> & dist_matrix, int max_remaining);
bool compare_transitions(struct Transition & first,
                         struct Transition & second);
template <class T>
std::vector <T> * vector_from_list(std::list <T> & list);
std::vector <std::vector <CompoundLoop>> * dp_table(
  std::list <struct Transition> & transition_list, int max_loop_length);
bool transition_ranges_overlap(std::list <struct Transition> transitions,
                               struct Transition primitive);
int compound_loop_length(struct CompoundLoop & loop);
struct CompoundLoop * merge_compound_loops(struct CompoundLoop & first,
                                           struct CompoundLoop & second);
void print_transitions_list(std::list <struct Transition> & transitions);
int transition_length(struct Transition transition);
void schedule_transitions(struct CompoundLoop & loop);
struct Transition remove_latest_transition(
  std::list <struct Transition> & transitions);
std::list <std::list <struct Transition>> * continuous_ranges(
  std::list <struct Transition> transitions);
std::vector <struct LoopFrame> * create_loop_frame_sequence(
  struct CompoundLoop & loop, std::string filename);
cv::Mat blend_frames(cv::Mat & frame_a, float alpha, cv::Mat & frame_b,
                     float beta);
cv::vector <cv::Mat> * blend_frames_at_transitions(
  std::vector <cv::Mat> & frames, std::list <int> & transition_points,
  int frames_per_blend);
int correct_frame_index(int sequence_length, int frame_index);
void save_parameters(std::string filepath);
void save_transition_pairs(std::vector <cv::Mat> & frames,
                           std::vector <struct LoopFrame> & sequence,
                           std::string filepath);
void setup_output_dirs(std::string filepath);
void save_matrix(std::vector <std::vector <double>> & matrix,
                 std::string filepath);
std::string output_name();
void save_distance_values(std::vector <std::vector <double>> & matrix,
                          std::string filepath);
std::vector <std::vector <double>> * load_distance_matrix(
  std::string filepath);
void load_parameters_from_string(std::string parameters);
std::list <std::string> load_parameters_from_file(std::string filename);
void save_images(std::vector <cv::Mat> & images, std::string filepath);
std::vector <cv::Mat> * blend_transitions(
  std::vector <struct LoopFrame> & frames, std::vector <float> weights);
std::vector <float> crossfade_weights(int num_frames_in_blend);
