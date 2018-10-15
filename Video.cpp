#include "vtg.h"

Video::Video()
{
  this->fps = 0;
  this->frame_width = 0;
  this->frame_height = 0;
} // function Video::Video

Video::Video(std::vector <cv::Mat> * frames, double fps, int frame_width,
             int frame_height)
{
  this->frames = *frames;
  this->fps = fps;
  this->frame_width = frame_width;
  this->frame_height = frame_height;
} // Video::Video

std::vector <cv::Mat> & Video::get_frames()
{
  return this->frames;
} // function Video::get_frames

int Video::get_frame_height() const
{
  return this->frame_height;
} // function Video::get_frame_height

int Video::get_frame_width() const
{
  return this->frame_width;
} // function Video::get_frame_width

double Video::get_fps() const
{
  return this->fps;
} // function Video::get_fps

void Video::set_frames(std::vector <cv::Mat> & frames)
{
  this->frames = frames;
} // function Video::set_frames

void Video::set_frame_height(int frame_height)
{
  this->frame_height = frame_height;
} // function Video::set_frame_height

void Video::set_frame_width(int frame_width)
{
  this->frame_width = frame_width;
} // function Video::set_frame_width

void Video::set_fps(double fps)
{
  this->fps = fps;
} // function Video::set_fps

bool Video::operator==(Video & other_video)
{
  std::vector <cv::Mat> ::iterator it_a;
  std::vector <cv::Mat> ::iterator it_b;
  for (it_a = this->frames.begin(), it_b = other_video.frames.begin();
       it_a != this->frames.end() && it_b != other_video.frames.end();
       it_a++, it_b++)
  {
    cv::Mat difference = *it_a != *it_b;
    if (cv::countNonZero(difference) != 0)
    {
      return false;
    } // if
  } // for

  return this->frame_height == other_video.frame_height
    && this->frame_width == other_video.frame_width
    && this->fps == other_video.fps;
} // function Video::operator==
