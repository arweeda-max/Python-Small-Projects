"""
Spider Cursor - CodeWithArweeda
Python + OpenCV + Inverse Kinematics

Install:   pip install opencv-python numpy
Run:       python spider_cursor.py
Quit:      press Q or ESC
"""

import math
import sys

import cv2
import numpy as np

W, H = 960, 600
BG = (14, 8, 10)  # dark background (BGR)

mouse = [W // 2, H // 2]


def on_mouse(event, x, y, flags, param):
    mouse[0], mouse[1] = x, y


# ---------------------------------------------------------------- vector helpers
def sub(a, b):
    return np.array([a[0] - b[0], a[1] - b[1]], dtype=float)


def add(a, b):
    return np.array([a[0] + b[0], a[1] + b[1]], dtype=float)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(v):
    n = math.hypot(v[0], v[1])
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    return t * t * (3 - 2 * t)


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- inverse kinematics
def two_bone(hip, foot, a, b, away_from):
    """Find the knee position of a 2-segment leg.
    hip -> knee has length a, knee -> foot has length b."""
    d = dist(hip, foot)
    d = max(1e-3, min(d, a + b - 0.01))
    u = unit(sub(foot, hip))
    cosA = (a * a + d * d - b * b) / (2 * a * d)
    A = math.acos(max(-1.0, min(1.0, cosA)))
    k1 = add(hip, rot(u, A) * a)
    k2 = add(hip, rot(u, -A) * a)
    # choose the knee that bends away from the body
    return k1 if dist(k1, away_from) > dist(k2, away_from) else k2


# ---------------------------------------------------------------- spider
class Spider:
    UPPER = 38  # upper leg length
    LOWER = 46  # lower leg length
    STEP_DIST = 46  # how far a foot can drift before stepping
    STEP_TIME = 0.11  # seconds for one step

    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.zeros(2)
        self.theta = 0.0
        self.trail = []

        # leg layout in body space (x = forward, y = sideways)
        hip_x = [20, 7, -7, -20]
        rest_x = [62, 34, -8, -50]
        rest_y = [58, 70, 70, 60]
        self.legs = []
        for side in (-1, 1):
            for i in range(4):
                self.legs.append(
                    {
                        "hip_off": np.array([hip_x[i], side * 9.0]),
                        "rest_off": np.array([rest_x[i], side * rest_y[i]]),
                        "foot": None,
                        "from": None,
                        "to": None,
                        "t": 1.0,
                        "stepping": False,
                        "group": (i + (0 if side < 0 else 1)) % 2,
                    }
                )
        self.reset_feet()

    # body-space -> world-space
    def to_world(self, off):
        return add(self.pos, rot(off, self.theta))

    def reset_feet(self):
        for lg in self.legs:
            lg["foot"] = self.to_world(lg["rest_off"])
            lg["to"] = lg["foot"].copy()
            lg["from"] = lg["foot"].copy()

    def update(self, dt, cur):
        d = sub(cur, self.pos)
        dd = math.hypot(d[0], d[1])

        # turn to face the cursor
        want = math.atan2(d[1], d[0])
        self.theta += angle_diff(want, self.theta) * min(1.0, dt * 7)

        # speed depends on distance to the cursor
        vtarget = min(300.0, max(0.0, (dd - 30) * 3.0))
        speed = math.hypot(self.vel[0], self.vel[1])
        speed += (vtarget - speed) * min(1.0, dt * 9)
        self.vel = np.array([math.cos(self.theta), math.sin(self.theta)]) * speed
        self.pos = self.pos + self.vel * dt

        self.trail.append(self.pos.copy())
        if len(self.trail) > 70:
            self.trail.pop(0)

        # legs
        for lg in self.legs:
            rest = self.to_world(lg["rest_off"])
            if lg["stepping"]:
                lg["t"] += dt / self.STEP_TIME
                if lg["t"] >= 1.0:
                    lg["t"] = 1.0
                    lg["stepping"] = False
                    lg["foot"] = lg["to"].copy()
                else:
                    lg["foot"] = lerp(lg["from"], lg["to"], ease(lg["t"]))
            else:
                other = [o for o in self.legs if o["group"] != lg["group"]]
                busy = any(o["stepping"] for o in other)
                far = dist(lg["foot"], rest)
                if far > self.STEP_DIST * 1.6 or (far > self.STEP_DIST and not busy):
                    lg["stepping"] = True
                    lg["t"] = 0.0
                    lg["from"] = lg["foot"].copy()
                    # step a bit ahead of the body so it looks natural
                    lg["to"] = add(rest, self.vel * self.STEP_TIME * 1.5)

    # ------------------------------------------------------------ drawing
    def draw(self, glow, sharp):
        leg_col = (255, 140, 200)  # BGR -> soft purple
        joint_col = (255, 230, 255)

        # trail (the silk thread behind the spider)
        if len(self.trail) > 2:
            pts = np.array(self.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(glow, [pts], False, (200, 120, 160), 2, cv2.LINE_AA)
            cv2.polylines(sharp, [pts], False, (230, 190, 220), 1, cv2.LINE_AA)

        # legs
        for lg in self.legs:
            hip = self.to_world(lg["hip_off"])
            foot = lg["foot"]
            reach = self.UPPER + self.LOWER - 2
            if dist(hip, foot) > reach:
                foot = add(hip, unit(sub(foot, hip)) * reach)
                lg["foot"] = foot
            knee = two_bone(hip, foot, self.UPPER, self.LOWER, self.pos)
            p = [tuple(map(int, q)) for q in (hip, knee, foot)]
            for img, th in ((glow, 6), (sharp, 2)):
                cv2.line(img, p[0], p[1], leg_col, th, cv2.LINE_AA)
                cv2.line(img, p[1], p[2], leg_col, th, cv2.LINE_AA)
            cv2.circle(sharp, p[1], 3, joint_col, -1, cv2.LINE_AA)
            cv2.circle(glow, p[2], 5, joint_col, -1, cv2.LINE_AA)

        # body: abdomen + head
        ang = math.degrees(self.theta)
        ab = tuple(map(int, self.to_world(np.array([-14.0, 0.0]))))
        hd = tuple(map(int, self.to_world(np.array([16.0, 0.0]))))
        cv2.ellipse(glow, ab, (30, 22), ang, 0, 360, (255, 90, 190), -1, cv2.LINE_AA)
        cv2.ellipse(sharp, ab, (26, 18), ang, 0, 360, (230, 110, 170), -1, cv2.LINE_AA)
        cv2.ellipse(sharp, ab, (26, 18), ang, 0, 360, (255, 190, 235), 1, cv2.LINE_AA)
        cv2.ellipse(glow, hd, (16, 13), ang, 0, 360, (255, 120, 200), -1, cv2.LINE_AA)
        cv2.ellipse(sharp, hd, (14, 11), ang, 0, 360, (240, 150, 210), -1, cv2.LINE_AA)

        # eyes + fangs
        for s in (-1, 1):
            eye = tuple(map(int, self.to_world(np.array([24.0, s * 5.0]))))
            fang = tuple(map(int, self.to_world(np.array([34.0, s * 4.0]))))
            cv2.circle(sharp, eye, 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(sharp, tuple(map(int, self.to_world(np.array([27.0, s * 4.0])))),
                     fang, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------- scene
def draw_cursor(glow, sharp, cur, t):
    r = 6 + int(2 * math.sin(t * 8))
    cv2.circle(glow, cur, r + 8, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(sharp, cur, r, (255, 255, 120), -1, cv2.LINE_AA)
    cv2.putText(sharp, f"({cur[0]}, {cur[1]})", (cur[0] + 14, cur[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 120), 1, cv2.LINE_AA)


def render(spider, cur, t):
    glow = np.zeros((H, W, 3), dtype=np.uint8)
    sharp = np.zeros((H, W, 3), dtype=np.uint8)

    spider.draw(glow, sharp)
    draw_cursor(glow, sharp, cur, t)

    glow = cv2.GaussianBlur(glow, (0, 0), 11)
    frame = np.full((H, W, 3), BG, dtype=np.uint8)
    frame = cv2.add(frame, (glow * 0.9).astype(np.uint8))
    frame = cv2.add(frame, sharp)

    cv2.putText(frame, "legs 8   eyes 8   fangs 2", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 110, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q = quit", (W - 90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 110, 100), 1, cv2.LINE_AA)
    return frame


def main():
    cv2.namedWindow("Spider Cursor")
    cv2.setMouseCallback("Spider Cursor", on_mouse)

    spider = Spider(W / 2, H / 2)
    tick = cv2.getTickCount
    freq = cv2.getTickFrequency()
    last = tick()
    t = 0.0

    while True:
        now = tick()
        dt = min((now - last) / freq, 0.05)
        last = now
        t += dt

        spider.update(dt, (mouse[0], mouse[1]))
        frame = render(spider, (mouse[0], mouse[1]), t)
        cv2.imshow("Spider Cursor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--test" in sys.argv:
        # headless test: simulate a moving cursor and save a few frames
        sp = Spider(W / 2, H / 2)
        out = []
        t = 0.0
        for i in range(240):
            cur = (int(W / 2 + 320 * math.cos(t * 1.1)), int(H / 2 + 190 * math.sin(t * 1.7)))
            sp.update(1 / 60, cur)
            t += 1 / 60
            if i in (40, 90, 150, 230):
                out.append(render(sp, cur, t))
        cv2.imwrite("test_frames.png", np.vstack([np.hstack(out[:2]), np.hstack(out[2:])]))
        print("ok")
    else:
        main()
