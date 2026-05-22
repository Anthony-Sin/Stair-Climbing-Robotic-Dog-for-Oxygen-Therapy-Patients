from nav_msgs.msg import OccupancyGrid

class PointXY:
    def __init__(self):
        self.x = 0
        self.y = 0

class PointXYZI:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.intensity = 0.0

class GridMap:
    def __init__(self):
        self.position_x = 0.0
        self.position_y = 0.0
        self.cell_size = 0.0
        self.length_x = 0.0
        self.length_y = 0.0
        self.cloud_in_topic = ""
        self.frame_out = ""
        self.mapi_topic_name = ""
        self.maph_topic_name = ""
        self.topleft_x = 0.0
        self.topleft_y = 0.0
        self.bottomright_x = 0.0
        self.bottomright_y = 0.0
        self.cell_num_x = 0
        self.cell_num_y = 0
        self.intensity_factor = 0.0
        self.height_factor = 0.0

    def initGrid(self, grid : OccupancyGrid):
        grid.header.frame_id = self.frame_out  # TODO
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 0.0
        grid.info.origin.orientation.x = 0.0
        grid.info.origin.orientation.y = 0.0
        grid.info.origin.orientation.z = 1.0
        grid.info.origin.position.x = self.position_x + self.length_x / 2.0
        grid.info.origin.position.y = self.position_y + self.length_y / 2.0
        grid.info.width = int(self.length_x / self.cell_size)
        grid.info.height = int(self.length_y / self.cell_size)
        grid.info.resolution = self.cell_size

    def paramRefresh(self):
        self.topleft_x = self.position_x + self.length_x / 2.0
        self.bottomright_x = self.position_x - self.length_x / 2.0
        self.topleft_y = self.position_y + self.length_y / 2.0
        self.bottomright_y = self.position_y - self.length_y / 2.0
        self.cell_num_x = int(self.length_x / self.cell_size)
        self.cell_num_y = int(self.length_y / self.cell_size)
        if self.cell_num_x > 0:
            print(
                "Cells: %d*%d px, subscribed to %s [%f, %f] [%f, %f]"
                % (
                    self.cell_num_x,
                    self.cell_num_y,
                    self.cloud_in_topic,
                    self.topleft_x,
                    self.topleft_y,
                    self.bottomright_x,
                    self.bottomright_y,
                )
            )

    def getSize(self):
        return self.cell_num_x * self.cell_num_y

    def getSizeX(self):
        return self.cell_num_x

    def getSizeY(self):
        return self.cell_num_y

    def getLengthX(self):
        return self.length_x

    def getLengthY(self):
        return self.length_y

    def getResolution(self):
        return self.cell_size
