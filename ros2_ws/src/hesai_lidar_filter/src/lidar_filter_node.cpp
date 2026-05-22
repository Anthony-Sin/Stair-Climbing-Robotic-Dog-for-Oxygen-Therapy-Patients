#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "DROH.h"

class LidarFilterNode : public rclcpp::Node {
public:
    LidarFilterNode() : Node("lidar_filter_node") {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/xt16/lidar_points", 10,
            std::bind(&LidarFilterNode::callback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_points_filter", 10);
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Use PCL::PointXYZI to preserve intensity data
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud_in);

        // --- Step 1: Apply CropBox filter to remove points within a 0.2x0.2m box ---
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZI>);
        
        pcl::CropBox<pcl::PointXYZI> box_filter;
        box_filter.setInputCloud(cloud_in);
        
        box_filter.setMin(Eigen::Vector4f(-0.2, -0.3, -2.0, 1.0f));
        box_filter.setMax(Eigen::Vector4f(0.2, 0.5, 2.0, 1.0f));
        box_filter.setNegative(true); // Keep points *outside* the box
        box_filter.filter(*cloud_cropped);


        // --- Step 2: Apply PassThrough filter to cut points beyond 3m ---
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_pass(new pcl::PointCloud<pcl::PointXYZI>);
        
        pcl::PassThrough<pcl::PointXYZI> pass;
        pass.setInputCloud(cloud_cropped);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-2, 2);
        pass.filter(*cloud_pass);

        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-2, 2);
        pass.filter(*cloud_pass);

        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.2, 3.0);
        pass.filter(*cloud_pass);

        // --- Step 3: Apply Dynamic Radius Outlier Removal (DROR) filter ---
        pcl::PointCloud<pcl::PointXYZI> cloud_filtered;
        dror_filter_.Filter(cloud_pass, cloud_filtered);
        
        // --- Step 4: Flip all point coordinates ---
        for (auto& point : cloud_filtered.points) {
            point.x = -point.x;
            point.y = -point.y;
            point.z = -point.z;
        }

        // --- Step 5: Publish the final filtered and flipped cloud ---
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(cloud_filtered, output);
        output.header = msg->header;
        pub_->publish(output);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    DROR dror_filter_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarFilterNode>());
    rclcpp::shutdown();
    return 0;
}