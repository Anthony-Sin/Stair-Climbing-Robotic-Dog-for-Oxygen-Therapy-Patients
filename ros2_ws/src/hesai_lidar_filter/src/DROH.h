#pragma once

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cmath>

/**
 * @brief Dynamic Radius Outlier Removal filter class
 */
class DROR
{
public:
    /**
     * @brief default constructor
     */
    DROR() = default;

    /**
     * @brief default destructor
     */
    ~DROR() = default;

    /**
     * @brief Method for setting radius multiplier
     * @param radius_multiplier default = 3. This should be greater than 1
     */
    void SetRadiusMultiplier(double radius_multiplier);

    /**
     * @brief Method for retrieving radius multiplier
     */
    double GetRadiusMultiplier();

    /**
     * @brief Method for setting azimuth angle
     * @param azimuth_angle default = 0.04 degrees
     */
    void SetAzimuthAngle(double azimuth_angle);

    /**
     * @brief Method for getting azimuth angle in degrees
     */
    double GetAzimuthAngle();

    /**
     * @brief Method for setting minimum number of neighbors
     * @param min_neighbors default = 3 (includes the point being filtered)
     */
    void SetMinNeighbors(double min_neighbors);

    /**
     * @brief Method for retrieving minimum number of neighbors
     */
    double GetMinNeighbors();

    /**
     * @brief Method for setting the minimum search radius
     * @param min_search_radius default = 0.04
     */
    void SetMinSearchRadius(double min_search_radius);

    /**
     * @brief Method for retrieving the minimum search radius
     */
    double GetMinSearchRadius();

    /**
     * @brief Method for applying the DROR filter
     * @param input_cloud cloud to be filtered
     * @param filtered_cloud output filtered cloud
     */
    template <typename PointT>
    void Filter(typename pcl::PointCloud<PointT>::Ptr &input_cloud,
                typename pcl::PointCloud<PointT> &filtered_cloud);

private:
    double radius_multiplier_{3.0};
    double azimuth_angle_rad_{0.04 * M_PI / 180.0};
    double min_neighbors_{3.0};
    double min_search_radius_{0.04};
};

// Template implementation
template <typename PointT>
void DROR::Filter(typename pcl::PointCloud<PointT>::Ptr &input_cloud,
                  typename pcl::PointCloud<PointT> &filtered_cloud)
{
    using KdTreePtr = typename pcl::KdTreeFLANN<PointT>::Ptr;
    
    // Clear points in output cloud
    filtered_cloud.clear();

    // Initialize kd search tree
    KdTreePtr kd_tree_(new pcl::KdTreeFLANN<PointT>());
    kd_tree_->setInputCloud(input_cloud);

    // Go over all points and check which doesn't have enough neighbors
    for (typename pcl::PointCloud<PointT>::iterator it = input_cloud->begin();
         it != input_cloud->end(); ++it)
    {
        float x_i = it->x;
        float y_i = it->y;
        float range_i = sqrt(pow(x_i, 2) + pow(y_i, 2));
        
        // Calculate dynamic search radius based on distance and azimuth angle
        float search_radius_dynamic = radius_multiplier_ * 2 * range_i * sin(azimuth_angle_rad_);

        // Ensure minimum search radius
        if (search_radius_dynamic < min_search_radius_)
        {
            search_radius_dynamic = min_search_radius_;
        }

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        // Find neighbors within dynamic radius
        int neighbors = kd_tree_->radiusSearch(*it, search_radius_dynamic, 
                                               pointIdxRadiusSearch,
                                               pointRadiusSquaredDistance);

        // Keep point if it has enough neighbors
        if (neighbors >= min_neighbors_)
        {
            filtered_cloud.push_back(*it);
        }
    }
}

