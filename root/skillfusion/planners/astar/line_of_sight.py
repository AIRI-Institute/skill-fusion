import numpy as np


class LineOfSight:
    def __init__(self):
        self.unknownIsObstacle = True


    def checkTraversability(self, x, y, grid_map, threshold=0.99):
        if x < 0 or y < 0 or x >= grid_map.shape[0] or y >= grid_map.shape[1]:
            return False
        if grid_map[x, y, 1] >= threshold:
            #print('Check traversability {}, {}: Obstacle'.format(x, y))
            return False
        if grid_map[x, y, 0] == 0 and self.unknownIsObstacle:
            #print('Check traversability {}, {}: Unknown'.format(x, y))
            return False
        #print('Check traversability {}, {}: True'.format(x, y))
        return True


    def checkLine(self, x1, y1, x2, y2, grid_map):
        if not self.checkTraversability(x1, y1, grid_map, threshold=0.49) or not self.checkTraversability(x2, y2, grid_map, threshold=0.49):
            return False

        dx = x2 - x1
        dy = y2 - y1
        
        sign_x = 1 if dx>0 else -1 if dx<0 else 0
        sign_y = 1 if dy>0 else -1 if dy<0 else 0
        
        if dx < 0: dx = -dx
        if dy < 0: dy = -dy
        
        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy
        
        x, y = x1, y1
        
        error, t = el/2, 0
        
        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            if not self.checkTraversability(x, y, grid_map, threshold=0.49):
                return False

        return True