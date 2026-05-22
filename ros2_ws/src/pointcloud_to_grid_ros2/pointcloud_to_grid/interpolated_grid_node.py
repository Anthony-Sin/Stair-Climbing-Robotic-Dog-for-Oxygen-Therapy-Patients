import copy
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, qos_profile_system_default
import math

#         _--"-.
#      .-"      "-.
#     |""--..      '-.
#     |      ""--..   '-.
#     |.-. .-".    ""--..".
#     |'./  -_'  .-.      |
#     |      .-. '.-'   .-'
#     '--..  '.'    .-  -.
#          ""--..   '_'   :
#                ""--..   |
#                      ""-'



class InterpolatedGridNode(Node):
    def __init__(self):
        super().__init__("pointcloud_to_grid_node")

        # Instantiate input occupancy grid
        self.swiss_grid = OccupancyGrid()

        # Instantiate output gridmap
        self.cheddar_grid = OccupancyGrid()

        self.declare_parameter("unresolved_grid", "/unresolved_grid")
        self.declare_parameter("interpolated_grid", "/interpolated_grid")

        self.unresolved_grid    = self.get_parameter("unresolved_grid").get_parameter_value().string_value
        self.interpolated_grid  = self.get_parameter("interpolated_grid").get_parameter_value().string_value

        # Force Message QOS
        adjusted_policy = qos_profile_system_default
        adjusted_policy.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # Set up Interpolated Map Publisher
        self.publish_cheddar = self.create_publisher(
            OccupancyGrid, self.interpolated_grid, adjusted_policy
        )

        # Set up Swiss Map Subscriber
        self.sub_pc2 = self.create_subscription(
            OccupancyGrid, self.unresolved_grid, self.interpolation_callback, 1
        )

    def interpolation_callback(self, msg : OccupancyGrid):
        self.swiss_grid = msg

        # Set the output map parameters to the same as the input ones
        self.cheddar_grid = copy.deepcopy(self.swiss_grid)
        self.cheddar_grid = self.swiss_grid
        self.cheddar_grid = copy.deepcopy(self.swiss_grid)

        # Extract width and length for ease of use
        self.cheddar_width    = self.cheddar_grid.info.width
        self.cheddar_length   = self.cheddar_grid.info.height

        # Erase the cheddar grid data
        self.cheddar_grid.data = []

        # For every point in the swiss grid test
        for point in range(0, len(self.swiss_grid.data)):
        
            # Run the Pseudo Mean filter
            self.cheddar_grid.data.append(self.pseudo_mean(point))

        # Adjust Grid Headers and set data
        now = self.get_clock().now().to_msg()

        self.cheddar_grid.header.stamp        = now
        self.cheddar_grid.header.frame_id     = msg.header.frame_id
        self.cheddar_grid.info.map_load_time  = now

        # Publish new grid
        self.publish_cheddar.publish(self.cheddar_grid)


    def pseudo_mean(self, point : int):
        num_valid   = 0
        cell_sum    = 0

        # Short circuit if the point has been written to before
        if self.swiss_grid.data[point] != -128:
            return self.swiss_grid.data[point]

        # For each y and x value in a 3x3 square centered around the point
        for y in range(-1, 2):
            for x in range(-1, 2):

                # Save the position of the current 2D point in the 1D array
                checkPoint  = point + y * self.cheddar_width + x

                # Check if the point is within bounds and if it has been written to
                if (self.isValid(checkPoint) and self.swiss_grid.data[checkPoint] != -128):
                    num_valid   += 1
                    cell_sum    += self.swiss_grid.data[checkPoint]

        # Don't interpolate if there isn't enough data
        if num_valid < 5:
            return -128
        
        # Take a basic average of all available points if there are enough
        else:
            cell_avg = math.floor(cell_sum/num_valid)
            return cell_avg
        
    def isValid(self, point):
        return point >= 0 and point < self.cheddar_length*self.cheddar_width


def main(args=None):
    rclpy.init(args=args)
    node = InterpolatedGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
