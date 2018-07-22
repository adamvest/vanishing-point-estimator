from skimage import feature, transform, io, filters
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import argparse, sys


class VanishingPointEstimatorOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--image_path', required=True, help="path to image to detect vps in")
        self.parser.add_argument('--num_vps', type=int, default=2, help="number of vanishing points to compute")
        self.parser.add_argument('--num_ransac_iter', type=int, default=500, help="number of iterations to run ransac")
        self.parser.add_argument('--inlier_angle', type=float, default=5.0, help="max angle to consider an edge an inlier")
        self.parser.add_argument('--use_length', type=int, default=1, help="whether to use length when determining edge strength")
        self.parser.add_argument('--vote_factor', type=float, default=5.0, help="factor to multiply angle by before computing vote")

    def parse(self):
        return self.parser.parse_args()


def get_bresenham_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)

    #rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    #swap start and end points if necessary and store swap state
    swapped = False

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    #recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    #calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    #iterate over bounding box generating points between start and end
    y = y1
    points = []

    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)

        if error < 0:
            y += ystep
            error += dx

    #reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()

    return points


def compute_length(start, end):
    return np.sqrt(np.square(start[0] - end[0]) + np.square(start[1] - end[1]))


def calculate_line_strength(line_points, img_grads, use_length):
    grad_sum, length_factor = 0.0, 1.0

    if use_length:
        length_factor = compute_length(line_points[0], line_points[-1])

    for x1, y1 in line_points:
        grad_sum += length_factor * img_grads[y1, x1]

    return grad_sum / len(line_points)


def sort_by_strength(locations, directions, strengths):
    sorted_locs, sorted_dirs, sorted_strs = [], [], []
    sorted_idxs = np.argsort(strengths)[::-1]

    for idx in sorted_idxs:
        sorted_locs.append(locations[idx])
        sorted_dirs.append(directions[idx])
        sorted_strs.append(strengths[idx])

    return np.array(sorted_locs), np.array(sorted_dirs), np.array(sorted_strs)


def compute_edgelets(gray_img, use_length):
    scharr_h = filters.scharr_h(gray_img)
    scharr_v = filters.scharr_v(gray_img)
    img_grad_mags = np.sqrt(np.square(scharr_h) + np.square(scharr_v))
    edges = feature.canny(gray_img, sigma=2)
    lines = transform.probabilistic_hough_line(edges, line_length=3, line_gap=2)
    locations, directions, strengths = [], [], []

    for p0, p1 in lines:
        line_points = get_bresenham_line(p0, p1)
        p0, p1 = np.hstack((np.array(p0), 1)), np.hstack((np.array(p1), 1))
        strengths.append(calculate_line_strength(line_points, img_grad_mags, use_length))
        directions.append(np.cross(p0, p1))
        locations.append((p0, p1))

    locations, directions, strengths = sort_by_strength(locations, directions, strengths)

    return (locations, directions, strengths)


def normalize_homogenous_line(line):
    norm_factor = np.sqrt(np.square(line[0]) + np.square(line[1]))

    if norm_factor == 0.0:
        return np.array([0, 0, 1])

    return line / norm_factor


def angle_between_homogenous_lines(line1, line2):
    angle_in_rads = np.arccos(np.clip((line1[0] * line2[0]) + (line1[1] * line2[1]), -1.0, 1.0))
    angle = np.rad2deg(angle_in_rads)
    return angle if angle <= 90.0 else 180 - angle


def modified_calc_vote(theta, factor):
    return np.square(np.cos(np.deg2rad(theta * factor)))


def compute_votes(edgelets, vanishing_point, args):
    locations, directions, strengths = edgelets
    votes = []

    for i in range(len(locations)):
        p0_to_vp = np.cross(locations[i][0], vanishing_point)
        p1_to_vp = np.cross(locations[i][1], vanishing_point)
        norm_p0_to_vp = normalize_homogenous_line(p0_to_vp)
        norm_p1_to_vp = normalize_homogenous_line(p1_to_vp)
        norm_edge_dir = normalize_homogenous_line(directions[i])
        theta1 = angle_between_homogenous_lines(norm_p0_to_vp, norm_edge_dir)
        theta2 = angle_between_homogenous_lines(norm_p1_to_vp, norm_edge_dir)
        theta = min(theta1, theta2)
        length_factor = 1.0

        if args.use_length:
            length_factor = compute_length(locations[i][0][:2], locations[i][1][:2])

        if theta <= args.inlier_angle:
            votes.append(length_factor * modified_calc_vote(theta, args.vote_factor))
        else:
            votes.append(0.0)

    return np.array(votes)


def is_coinciding(line1, line2):
    intersect = np.cross(line1, line2)
    return np.array_equal(intersect, np.zeros(3))


def ransac_vanishing_point(edgelets, args):
    print("\nEstimating New Vanishing Point\n")
    _, directions, _ = edgelets
    num_pts = directions.shape[0]
    best_vp, best_votes, best_sum = None, None, 0.0
    top_20_per_idx, top_50_per_idx = num_pts // 5, num_pts // 2

    for ransac_iter in range(1, args.num_ransac_iter + 1):
        idx1 = np.random.choice(top_20_per_idx)
        idx2 = np.random.choice(top_50_per_idx)

        if idx1 == idx2 or is_coinciding(directions[idx1], directions[idx2]):
            continue

        current_vp = np.cross(directions[idx1], directions[idx2])
        current_votes = compute_votes(edgelets, current_vp, args)
        current_sum = current_votes.sum()

        if current_sum > best_sum:
            best_vp = current_vp
            best_votes = current_votes
            best_sum = current_sum
            print("Current best model has {} votes at iteration {}".format(
                best_sum, ransac_iter))

    return best_vp, best_votes


def remove_inliers(edgelets, votes):
    locations, directions, strengths = edgelets
    inliers = votes > 0
    not_inliers = ~inliers
    inlier_edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    remaining_edgelets = (locations[not_inliers], directions[not_inliers], strengths[not_inliers])
    return remaining_edgelets, inlier_edgelets


def get_vanishing_points_and_inliers(args):
    image = io.imread(args.image_path, as_grey=True)
    vps, inlier_sets = [], []
    edgelets = compute_edgelets(image, args.use_length)

    for i in range(args.num_vps):
        vp, votes = ransac_vanishing_point(edgelets, args)
        edgelets, inliers = remove_inliers(edgelets, votes)
        vps.append(vp)
        inlier_sets.append(inliers)

    return vps, inlier_sets


def visualize_inliers(image_path, inliers):
    image = io.imread(image_path, as_grey=True)
    edges = feature.canny(image, sigma=2)
    lines = transform.probabilistic_hough_line(edges, line_length=3, line_gap=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(edges * 0)
    ax[0].set_title('Input image')

    ax[1].imshow(edges * 0)

    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))

    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_title('Hough Lines')

    ax[2].imshow(edges * 0)
    inlier_colors = ["blue", "red", "green"]

    for i in range(len(inliers)):
        locations, _, _ = inliers[i]

        for p0, p1 in locations:
            x1, y1 = p0[0], p0[1]
            x2, y2 = p1[0], p1[1]
            ax[2].plot((x1, x2), (y1, y2), color=inlier_colors[i])

    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Inlier Sets')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable(adjustable="box")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = VanishingPointEstimatorOptions().parse()
    vps, inliers = get_vanishing_points_and_inliers(args)

    for i in range(len(vps)):
        print("\nHomogenous Vanishing Point:", vps[i])

        if vps[i][2] != 0:
            x = vps[i][0] / vps[i][2]
            y = vps[i][1] / vps[i][2]
            print("Cartesian Vanishing Point:", (x,y))
        else:
            print("Vanishing point is at infinity!")

        print("Number of Inlier Edges:", len(inliers[i][0]))

    visualize_inliers(args.image_path, inliers)
