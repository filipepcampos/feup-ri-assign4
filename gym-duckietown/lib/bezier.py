import numpy as np

CURVE_RIGHT_BEZIER = np.array([[[-0.20, 0, -0.50],[-0.20, 0, 0.00], [0.00, 0, 0.20], [0.50, 0, 0.20],],
                        [[0.2, 0, -0.50], [0.2, 0, -0.30], [0.3, 0, -0.15], [0.5, 0, -0.25]]])

# original
CURVE_LEFT_BEZIER = np.array([[[-0.20, 0, -0.50],[-0.20, 0, -0.20], [-0.30, 0, -0.20], [-0.50, 0, -0.10],],
                        [[0.2, 0, -0.50], [0.3, 0, 0.0], [-0.3, 0, 0.20], [-0.5, 0, 0.2]]])

TURN_LEFT_STEPS = 275
TURN_RIGHT_STEPS = 200
GO_STRAIGHT_STEPS = 100

avg_curve = lambda x, y: np.mean([x, y], axis=0)
avg_weighted_curve = lambda x, y, w: np.average([x, y], axis=0, weights=[w, 1 - w])

    
def bezier_curve(curve, timesteps):
    p0, p1, p2, p3 = curve[0], curve[1], curve[2], curve[3]
    p = lambda t: (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return np.array([p(t) for t in np.linspace(0, 1, timesteps)])


def bezier_curve_with_straight_ending(curve, timesteps, direction="left"):
    # describe the curve until 0.8 of the total time
    curve_timesteps = int(0.8 * timesteps)
    curve_points = bezier_curve(curve, curve_timesteps)

    # describe a straight line in 0.2 of the total time 
    first_point = curve_points[-1]
    direction = -1 if direction == "left" else 1
    straight_points = np.array([(first_point[0] + direction * i, 0, first_point[2]) for i in range(timesteps - curve_timesteps)])

    return np.concatenate((curve_points, straight_points))


def get_left_curve():
    return bezier_curve_with_straight_ending(CURVE_LEFT_BEZIER[0], TURN_LEFT_STEPS, "left")


def get_right_curve():
    return bezier_curve(CURVE_RIGHT_BEZIER[0], TURN_RIGHT_STEPS)

left_curve = get_left_curve()
right_curve = get_right_curve()