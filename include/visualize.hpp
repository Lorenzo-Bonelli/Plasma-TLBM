// visualize.hpp
#pragma once

#include "utils.hpp"

#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace visualize {

static constexpr int P = 9;
//---------------------------------------------------------------------------
// Initialization Update and Finalization
//---------------------------------------------------------------------------

// Prepare sample_points and allocate all ts_* buffers for T steps
void InitVisualization(const int NX,const int NY, const int T);

// Record one time-step worth of data into in-memory buffers + render frames
void UpdateVisualization(const int t, const int NX, const int NY,
    const std::vector<double>& ux_e,  const std::vector<double>& uy_e,
    const std::vector<double>& ux_i,  const std::vector<double>& uy_i,
    const std::vector<double>& ux_n,  const std::vector<double>& uy_n,
    const std::vector<double>& T_e,   const std::vector<double>& T_i,
    const std::vector<double>& T_n,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i,
    const std::vector<double>& rho_n, const std::vector<double>& rho_q,
    const std::vector<double>& Ex,    const std::vector<double>& Ey);


// After the run, plot all time-series buffers to PNG and release videos
void CloseVisualization();

//---------------------------------------------------------------------------
// Internal plotting helper
//---------------------------------------------------------------------------

// Plot one 2D time-series buffer into a PNG image using OpenCV
cv::Mat PlotTimeSeriesWithOpenCV(const std::vector<std::array<double,P>>& data,
                                 const std::string& title);
//---------------------------------------------------------------------------
// Visualization routines
//---------------------------------------------------------------------------

// Render density and charge maps into the density video stream
void VisualizationDensity(const int NX, const int NY,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i,
    const std::vector<double>& rho_q);

// Render velocity components and magnitude into the velocity video stream
void VisualizationVelocity(const int NX, const int NY,
    const std::vector<double>& ux_e,  const std::vector<double>& uy_e,
    const std::vector<double>& ux_i,  const std::vector<double>& uy_i);

// Render electron and ion temperature maps into the temperature video stream
void VisualizationTemperature(const int NX, const int NY,
    const std::vector<double>& T_e,   const std::vector<double>& T_i, const std::vector<double>& T_n);

//---------------------------------------------------------------------------
// Helper for the Videos
//---------------------------------------------------------------------------
cv::Mat normalize_and_color(const cv::Mat& src, const double vmin, const double vmax);
cv::Mat wrap_with_label(const cv::Mat& img, const std::string& label);

} // namespace visualize
