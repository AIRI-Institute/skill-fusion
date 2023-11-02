import numpy as np


class LineOfSight:
    def __init__(self, agent_size=1):
        self.agent_size = agent_size
        self.cells = []
        for x in range(-agent_size, agent_size + 1):
            for y in range(-agent_size, agent_size + 1):
                if x ** 2 + y ** 2 <= agent_size ** 2:
                    self.cells.append((x, y))
        self.unknownIsObstacle = True


    def checkTraversability(self, x, y, grid_map):
        for dx, dy in self.cells:
            if x + dx < 0 or y + dy < 0 or x + dx >= grid_map.shape[0] or y + dy >= grid_map.shape[1]:
                return False
            if grid_map[x + dx, y + dy, 1] > 0:
                return False
            if grid_map[x + dx, y + dy, 0] == 0 and self.unknownIsObstacle:
                return False
        return True


    def checkLine(self, x1, y1, x2, y2, grid_map):
        if not self.checkTraversability(x1, y1, grid_map) or not self.checkTraversability(x2, y2, grid_map):
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
            if not self.checkTraversability(x, y, grid_map):
                return False

        return True