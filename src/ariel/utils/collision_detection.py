import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class AABB:
    def __init__(self, x, y, w, h, obj_id=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.obj_id = obj_id

    def intersects(self, other) -> bool:
        return not (
            self.x + self.w < other.x - other.w or
            self.x - self.w > other.x + other.w or
            self.y + self.h < other.y - other.h or
            self.y - self.h > other.y + other.h
        )

    def inside(self, boundary) -> bool:
        return (
            self.x - self.w >= boundary.x - boundary.w and
            self.x + self.w <= boundary.x + boundary.w and
            self.y - self.h >= boundary.y - boundary.h and
            self.y + self.h <= boundary.y + boundary.h
        )


class Quadtree:
    def __init__(self, boundary: AABB, capacity=4, depth=0, max_depth=6):
        self.boundary = boundary
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.objects: list[AABB] = []
        self.divided = False

    def subdivide(self):
        hw, hh = self.boundary.w / 2, self.boundary.h / 2
        x, y = self.boundary.x, self.boundary.y

        self.nw = Quadtree(AABB(x - hw, y - hh, hw, hh), self.capacity, self.depth+1, self.max_depth)
        self.ne = Quadtree(AABB(x + hw, y - hh, hw, hh), self.capacity, self.depth+1, self.max_depth)
        self.sw = Quadtree(AABB(x - hw, y + hh, hw, hh), self.capacity, self.depth+1, self.max_depth)
        self.se = Quadtree(AABB(x + hw, y + hh, hw, hh), self.capacity, self.depth+1, self.max_depth)

        self.divided = True

    def insert(self, box: AABB) -> bool:
        if not box.inside(self.boundary):
            return False

        if len(self.objects) < self.capacity or self.depth >= self.max_depth:
            self.objects.append(box)
            return True

        if not self.divided:
            self.subdivide()

        return (
            self.nw.insert(box) or
            self.ne.insert(box) or
            self.sw.insert(box) or
            self.se.insert(box)
        )

    def query(self, range_box: AABB, found=None):
        if found is None:
            found = []

        if not self.boundary.intersects(range_box):
            return found

        for obj in self.objects:
            if obj.intersects(range_box):
                found.append(obj)

        if self.divided:
            self.nw.query(range_box, found)
            self.ne.query(range_box, found)
            self.sw.query(range_box, found)
            self.se.query(range_box, found)

        return found

    def draw(self, ax):
        rect = patches.Rectangle(
            (self.boundary.x - self.boundary.w, self.boundary.y - self.boundary.h),
            self.boundary.w * 2,
            self.boundary.h * 2,
            linewidth=0.5,
            edgecolor="blue",
            facecolor="none"
        )
        ax.add_patch(rect)

        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.sw.draw(ax)
            self.se.draw(ax)


if __name__ == "__main__":
    np.random.seed(123)

    boundary = AABB(0, 0, 1000, 1000)
    qt = Quadtree(boundary)

    robots = []
    num_boxes = 1000
    max_attempts = 1000

    for i in range(num_boxes):
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            x, y = np.random.uniform(-990, 990, size=2)
            new_box = AABB(x, y, 2, 2, obj_id=i)

            # Check collision before inserting
            candidates = qt.query(new_box)
            if not any(new_box.intersects(c) for c in candidates):
                qt.insert(new_box)
                robots.append(new_box)
                placed = True
            attempts += 1

    print(f"Placed {len(robots)} boxes without collisions")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect("equal")

    qt.draw(ax)

    for box in robots:
        rect = patches.Rectangle(
            (box.x - box.w, box.y - box.h),
            box.w * 2,
            box.h * 2,
            linewidth=1,
            edgecolor="black",
            facecolor="lightgreen"
        )
        ax.add_patch(rect)

    plt.show()
