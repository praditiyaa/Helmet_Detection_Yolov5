import math


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + w) // 2
            cy = (y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            assigned_id = None

            for id, pt in self.center_points.items():
                if id in objects_bbs_ids:
                    continue

                dist = math.hypot(cx - pt[0], cy - pt[1])
                print(dist)

                if dist < 90:
                    assigned_id = id
                    same_object_detected = True
                    break

            # New object is detected, assign the ID to that object
            if same_object_detected:
                self.center_points[assigned_id] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, assigned_id])
            else:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {obj_bb_id[4]: self.center_points[obj_bb_id[4]] for obj_bb_id in objects_bbs_ids}
        self.center_points = new_center_points

        return objects_bbs_ids
