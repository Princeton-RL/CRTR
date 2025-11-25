"""
This file is part of the Magic Cube project.

license
-------
Copyright 2012 David W. Hogg (NYU).

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301 USA.

usage
-----
- initialize a solved cube with `c = Cube(N)` where `N` is the side length.
- randomize a cube with `c.randomize(32)` where `32` is the number of random moves to make.
- make cube moves with `c.move()` and turn the whole cube with `c.turn()`.
- make figures with `c.render().savefig(fn)` where `fn` is the filename.
- change sticker colors with, eg, `c.stickercolors[c.colordict["w"]] = "k"`.

conventions
-----------
- This is a model of where the stickers are, not where the solid cubies are.  That's a bug not a feature.
- Cubes are NxNxN in size.
- The faces have integers and one-letter names. The one-letter face names are given by the dictionary `Cube.facedict`.
- The layers of the cube have names that are composed of a face letter and a number, with 0 indicating the outermost face.
- Every layer has two layer names, for instance, (F, 1) and (B, 1) are the same layer of a 3x3x3 cube; (F, 1) and (B, 3) are the same layer of a 5x5x5.
- The colors have integers and one-letter names. The one-letter color names are given by the dictionary `Cube.colordict`.
- Convention is x before y in face arrays, plus an annoying baked-in left-handedness.  Sue me.  Or fork, fix, pull-request.

to-do
-----
- Write translations to other move languages, so you can take a string of moves from some website (eg, <http://www.speedcubing.com/chris/3-permutations.html>) and execute it.
- Keep track of sticker ID numbers and orientations to show that seemingly unchanged parts of big cubes have had cubie swaps or stickers rotated.
- Figure out a physical "cubie" model to replace the "sticker" model.

"""
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import torch

class Actions(Enum):
    U = {"name": "U", "f": "U", "d": 1, "opposite": "U_1"},
    U_1 = {"name": "U'", "f": "U", "d": -1, "opposite": "U"},
    D = {"name": "D", "f": "D", "d": 1, "opposite": "D_1"},
    D_1 = {"name": "D'", "f": "D", "d": -1, "opposite": "D"},
    F = {"name": "F", "f": "F", "d": 1, "opposite": "F_1"},
    F_1 = {"name": "F'", "f": "F", "d": -1, "opposite": "F"},
    B = {"name": "B", "f": "B", "d": 1, "opposite": "B_1"},
    B_1 = {"name": "B'", "f": "B", "d": -1, "opposite": "B"},
    R = {"name": "R", "f": "R", "d": 1, "opposite": "R_1"},
    R_1 = {"name": "R'", "f": "R", "d": -1, "opposite": "R"},
    L = {"name": "L", "f": "L", "d": 1, "opposite": "L_1"},
    L_1 = {"name": "L''", "f": "L", "d": -1, "opposite": "L"},
    U_2 = {"name": "U''", "f": "U", "d": 2, "opposite": "U_2"},
    D_2 = {"name": "D''", "f": "D", "d": 2, "opposite": "D_2"},
    F_2 = {"name": "F''", "f": "F", "d": 2, "opposite": "F_2"},
    B_2 = {"name": "B''", "f": "B", "d": 2, "opposite": "B_2"},
    R_2 = {"name": "R''", "f": "R", "d": 2, "opposite": "R_2"},
    L_2 = {"name": "L''", "f": "L", "d": 2, "opposite": "L_2"},

def reverse_cube_labels():
    return {'y':0, 'w':1, 'r':2, 'o':3, 'g':4, 'b':5}

def cube_labels():
    return 'ywrogb'

def cube_str_to_bin(string_obs):
    assert len(string_obs) == 54

    stickers = [reverse_cube_labels()[x] for x in string_obs]
    indexes = np.eye(6)[stickers]
    faces = indexes.reshape((6, 3, 3, 6))
    aligned_faces = np.array([np.rot90(face, k=-1, axes=(0, 1)) for face in faces])
    ordered_faces = [aligned_faces[i] for i in [0, 5, 2, 4, 3, 1]]

    return np.array(ordered_faces)


def cube_str_to_state(string_obs):
    return np.argmax(cube_str_to_bin(string_obs), axis=-1)

class Cube(object):
    """
    Cube
    ----
    Initialize with arguments:
    - `N`, the side length (the cube is `N`x`N`x`N`)
    - optional `whiteplastic=True` if you like white cubes
    """

    action_names = [a.name for a in Actions]

    facedict = {"U": 0, "D": 1, "F": 2, "B": 3, "R": 4, "L": 5}
    dictface = dict([(v, k) for k, v in facedict.items()])
    normals = [np.array([0., 1., 0.]), np.array([0., -1., 0.]),
               np.array([0., 0., 1.]), np.array([0., 0., -1.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.])]
    # this xdirs has to be synchronized with the self.move() function
    xdirs = [np.array([1., 0., 0.]), np.array([1., 0., 0.]),
             np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
             np.array([0., 0., -1.]), np.array([0, 0., 1.])]
    colordict = {"w": 0, "y": 1, "b": 2, "g": 3, "o": 4, "r": 5}
    pltpos = [(0., 1.05), (0., -1.05), (0., 0.), (2.10, 0.), (1.05, 0.), (-1.05, 0.)]
    labelcolor = "#7f00ff"

    def __init__(self, N, stickers=None, whiteplastic=False):
        """
        (see above)
        """
        self.N = N

        if stickers is None:
            self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        else:
            mapping = {'o': 0, 'y': 1, 'r': 2, 'b': 3, 'g': 4, 'w': 5}
            reverse_mapping = {v: k for k, v in mapping.items()}
            if not (type(stickers[0]) == str or type(stickers[0]) == np.str_):
                # if element type is not int, take .item()
                if type(stickers[0]) == torch.Tensor:
                    stickers = [reverse_mapping[i.item()] for i in stickers]
                else:
                    stickers = [reverse_mapping[i] for i in stickers]

            stickers = ''.join(stickers)
            self.stickers = cube_str_to_state(stickers)

        self.stickercolors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        self.stickerthickness = 0.001  # sticker thickness in units of total cube size
        self.stickerwidth = 0.9  # sticker size relative to cubie size (must be < 1)
        if whiteplastic:
            self.plasticcolor = "#dfdfdf"
        else:
            self.plasticcolor = "#1f1f1f"
        self.fontsize = 12. * (self.N / 5.)
        self.solved_score = self.score()

    def turn(self, f, d):
        """
        Turn whole cube (without making a layer move) around face `f`
        `d` 90-degree turns in the clockwise direction.  Use `d=3` or
        `d=-1` for counter-clockwise.
        """
        for l in range(self.N):
            self.move(f, l, d)
        return None

    def move(self, f, l, d):
        """
        Make a layer move of layer `l` parallel to face `f` through
        `d` 90-degree turns in the clockwise direction.  Layer `0` is
        the face itself, and higher `l` values are for layers deeper
        into the cube.  Use `d=3` or `d=-1` for counter-clockwise
        moves, and `d=2` for a 180-degree move..
        """
        i = self.facedict[f]
        l2 = self.N - 1 - l
        assert l < self.N
        ds = range((d + 4) % 4)
        if f == "U":
            f2 = "D"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["F"], range(self.N), l2),
                              (self.facedict["R"], range(self.N), l2),
                              (self.facedict["B"], range(self.N), l2),
                              (self.facedict["L"], range(self.N), l2)])
        if f == "D":
            return self.move("U", l2, -d)
        if f == "F":
            f2 = "B"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], range(self.N), l),
                              (self.facedict["L"], l2, range(self.N)),
                              (self.facedict["D"], range(self.N)[::-1], l2),
                              (self.facedict["R"], l, range(self.N)[::-1])])
        if f == "B":
            return self.move("F", l2, -d)
        if f == "R":
            f2 = "L"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], l2, range(self.N)),
                              (self.facedict["F"], l2, range(self.N)),
                              (self.facedict["D"], l2, range(self.N)),
                              (self.facedict["B"], l, range(self.N)[::-1])])
        if f == "L":
            return self.move("R", l2, -d)
        for d in ds:
            if l == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if l == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)
        # print("moved", f, l, len(ds))
        return None

    def _rotate(self, args):
        """
        Internal function for the `move()` function.
        """
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def randomize(self, number):
        """
        Make `number` randomly chosen moves to scramble the cube.
        """
        for t in range(number):
            f = self.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            d = 1 + np.random.randint(3)
            self.move(f, l, d)
        return None

    def _render_points(self, points, viewpoint):
        """
        Internal function for the `render()` function.  Clunky
        projection from 3-d to 2-d, but also return a zorder variable.
        """
        v2 = np.dot(viewpoint, viewpoint)
        zdir = viewpoint / np.sqrt(v2)
        xdir = np.cross(np.array([0., 1., 0.]), zdir)
        xdir /= np.sqrt(np.dot(xdir, xdir))
        ydir = np.cross(zdir, xdir)
        result = []
        for p in points:
            dpoint = p - viewpoint
            dproj = 0.5 * dpoint * v2 / np.dot(dpoint, -1. * viewpoint)
            result += [np.array([np.dot(xdir, dproj),
                                 np.dot(ydir, dproj),
                                 np.dot(zdir, dpoint / np.sqrt(v2))])]
        return result

    def render_views(self, ax):
        """
        Make three projected 3-dimensional views of the cube for the
        `render()` function.  Because of zorder / occulting issues,
        this code is very brittle; it will not work for all viewpoints
        (the `np.dot(zdir, viewpoint)` test is not general; the corect
        test involves the "handedness" of the projected polygon).
        """
        csz = 2. / self.N
        x2 = 8.
        x1 = 0.5 * x2
        for viewpoint, shift in [(np.array([-x1, -x1, x2]), np.array([-1.5, 3.])),
                                 (np.array([x1, x1, x2]), np.array([0.5, 3.])),
                                 (np.array([x2, x1, -x1]), np.array([2.5, 3.]))]:
            for f, i in self.facedict.items():
                zdir = self.normals[i]
                if np.dot(zdir, viewpoint) < 0:
                    continue
                xdir = self.xdirs[i]
                ydir = np.cross(zdir, xdir)  # insanity: left-handed!
                psc = 1. - 2. * self.stickerthickness
                corners = [psc * zdir - psc * xdir - psc * ydir,
                           psc * zdir + psc * xdir - psc * ydir,
                           psc * zdir + psc * xdir + psc * ydir,
                           psc * zdir - psc * xdir + psc * ydir]
                projects = self._render_points(corners, viewpoint)
                xys = [p[0:2] + shift for p in projects]
                zorder = np.mean([p[2] for p in projects])
                ax.add_artist(Polygon(xys, ec="none", fc=self.plasticcolor))
                for j in range(self.N):
                    for k in range(self.N):
                        corners = self._stickerpolygon(xdir, ydir, zdir, csz, j, k)
                        projects = self._render_points(corners, viewpoint)
                        xys = [p[0:2] + shift for p in projects]
                        ax.add_artist(Polygon(xys, ec="none", fc=self.stickercolors[self.stickers[i, j, k]]))
                x0, y0, zorder = self._render_points([1.5 * self.normals[i], ], viewpoint)[0]
                # ax.text(x0 + shift[0], y0 + shift[1], f, color=self.labelcolor,
                #         ha="center", va="center", rotation=20, fontsize=self.fontsize / (-zorder))
        return None

    def _stickerpolygon(self, xdir, ydir, zdir, csz, j, k):
        small = 0.5 * (1. - self.stickerwidth)
        large = 1. - small
        return [zdir - xdir + (j + small) * csz * xdir - ydir + (k + small + small) * csz * ydir,
                zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + small) * csz * ydir,
                zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + small) * csz * ydir,
                zdir - xdir + (j + large) * csz * xdir - ydir + (k + small + small) * csz * ydir,
                zdir - xdir + (j + large) * csz * xdir - ydir + (k + large - small) * csz * ydir,
                zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + large) * csz * ydir,
                zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + large) * csz * ydir,
                zdir - xdir + (j + small) * csz * xdir - ydir + (k + large - small) * csz * ydir]

    def render_flat(self, ax):
        """
        Make an unwrapped, flat view of the cube for the `render()`
        function.  This is a map, not a view really.  It does not
        properly render the plastic and stickers.
        """
        for f, i in self.facedict.items():
            x0, y0 = self.pltpos[i]
            cs = 1. / self.N
            for j in range(self.N):
                for k in range(self.N):
                    ax.add_artist(Rectangle((x0 + j * cs, y0 + k * cs), cs, cs, ec=self.plasticcolor,
                                            fc=self.stickercolors[self.stickers[i, j, k]]))
            ax.text(x0 + 0.5, y0 + 0.5, f, color=self.labelcolor,
                    ha="center", va="center", rotation=20, fontsize=self.fontsize)
        return None

    def render(self, fig, flat=True, views=True):
        """
        Visualize the cube in a standard layout, including a flat,
        unwrapped view and three perspective views.
        """
        assert flat or views
        xlim = (-2.4, 3.4)
        ylim = (-1.2, 4.)
        if not flat:
            ylim = (2., 4.)
        if not views:
            xlim = (-1.2, 3.2)
            ylim = (-1.2, 2.2)
        if not fig:
            fig = plt.figure(figsize=((xlim[1] - xlim[0]) * self.N / 5., (ylim[1] - ylim[0]) * self.N / 5.))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False, xticks=[], yticks=[])
        if views:
            self.render_views(ax)
        if flat:
            self.render_flat(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig

    def score(self):
        """
        Calculate cube distance from solution
        """
        temp_score = 1
        for i in range(6):
            side = self.stickers[i]
            side_color = side[1][1]
            side_score = 0
            for x in range(3):
                for y in range(3):
                    if side[x][y] == side_color:
                        side_score += 1
            temp_score *= side_score
        return temp_score

    def move_by_action(self, action):
        # action = self.actions.get(action_name)
        f = action.value[0].get("f")
        d = action.value[0].get("d")
        self.move(f, 0, d)

    def solved(self, score):
        return score == self.solved_score

    def get_state(self):
        return self.stickers

    def opposite_actions(self, previous_action_name, action):
        return previous_action_name == action.value[0].get("opposite")


def checkerboard(cube):
    """
    Dumbness.
    """
    ls = range(cube.N)[::2]
    for f in ["U", "F", "R"]:
        for l in ls:
            cube.move(f, l, 2)
    if cube.N % 2 == 0:
        for l in ls:
            cube.move("F", l, 2)
    return None


if __name__ == "__main__":
    """
    Functional testing.
    """
    np.random.seed(42)
    c = Cube(3, whiteplastic=False)
    # c.turn("U", 1)
    # c.move("U", 0, -1)
    # # swap_off_diagonal(c, "R", 2, 1)
    # c.move("U", 0, 1)
    # # swap_off_diagonal(c, "R", 3, 2)
    # # checkerboard(c)

    # plt.ion()
    # plt.show()
    states = [
        "yyryyryyrbbbbbbbbbrrwrrwrrwgggggggggyooyooyoowwowwoww", 
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
        # "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
       
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"
    ]
    
    # {
    #  'yyyyyyyyybbbbbbooorrrrrrbbbggggggrrroooooogggwwwwwwwww', 
    #  'yyyyyyyyyrrrbbbbbbgggrrrrrroooggggggbbboooooowwwwwwwww', 
    #  'oyyoyyoyybbbbbbbbbyrryrryrrgggggggggoowoowoowrwwrwwrww', 
    #  'yyyyyyyyybbbbbbrrrrrrrrrgggggggggooooooooobbbwwwwwwwww', 
    #  'gggyyyyyyybbybbybbrrrrrrrrrggwggwggwooooooooowwwwwwbbb', 
    #  'yyryyryyrbbbbbbbbbrrwrrwrrwgggggggggyooyooyoowwowwowwo', 
    #  'yyyyyyyyyooobbbbbbbbbrrrrrrrrrgggggggggoooooowwwwwwwww', 
    #  'ryyryyryybbbbbbbbbwrrwrrwrrgggggggggooyooyooyowwowwoww', 
    #  'yyyyyygggbbybbybbyrrrrrrrrrwggwggwggooooooooobbbwwwwww', 
    #  'yyoyyoyyobbbbbbbbbrryrryrrygggggggggwoowoowoowwrwwrwwr', 
    #  'yyyyyybbbbbwbbwbbwrrrrrrrrryggyggyggooooooooogggwwwwww', 
    #  'bbbyyyyyywbbwbbwbbrrrrrrrrrggyggyggyooooooooowwwwwwggg'
    #  }
    
    states = [
     'yyyyyyyyybbbbbbooorrrrrrbbbggggggrrroooooogggwwwwwwwww', 
     'yyyyyyyyyrrrbbbbbbgggrrrrrroooggggggbbboooooowwwwwwwww', 
     'oyyoyyoyybbbbbbbbbyrryrryrrgggggggggoowoowoowrwwrwwrww', 
     'yyyyyyyyybbbbbbrrrrrrrrrgggggggggooooooooobbbwwwwwwwww', 
     'gggyyyyyyybbybbybbrrrrrrrrrggwggwggwooooooooowwwwwwbbb', 
     'yyryyryyrbbbbbbbbbrrwrrwrrwgggggggggyooyooyoowwowwowwo', 
     'yyyyyyyyyooobbbbbbbbbrrrrrrrrrgggggggggoooooowwwwwwwww', 
     'ryyryyryybbbbbbbbbwrrwrrwrrgggggggggooyooyooyowwowwoww', 
     
     'yyyyyygggbbybbybbyrrrrrrrrrwggwggwggooooooooobbbwwwwww',
     
     'yyoyyoyyobbbbbbbbbrryrryrrygggggggggwoowoowoowwrwwrwwr', 
     'yyyyyybbbbbwbbwbbwrrrrrrrrryggyggyggooooooooogggwwwwww', 
     'bbbyyyyyywbbwbbwbbrrrrrrrrrggyggyggyooooooooowwwwwwggg'
    ]
    
    solved_states = [
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
        "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww",
    ]
    
    perms = [[[15, 16, 17, 42, 43, 44, 33, 34, 35, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51, 52, 53], [42, 43, 44, 33, 34, 35, 24, 25, 26, 15, 16, 17, 51, 48, 45, 52, 49, 46, 53, 50, 47]],
    [[9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 0, 1, 2, 3, 4, 5, 6, 7, 8] , [18, 19, 20, 27, 28, 29, 36, 37, 38, 9, 10, 11, 6, 3, 0, 7, 4, 1, 8, 5, 2]],
    [[0, 3, 6, 38, 41, 44, 45, 48, 51, 18, 21, 24, 9, 10, 11, 12, 13, 14, 15, 16, 17] , [44, 41, 38, 51, 48, 45, 18, 21, 24, 0, 3, 6, 15, 12, 9, 16, 13, 10, 17, 14, 11]],
    [[15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] , [24, 25, 26, 33, 34, 35, 42, 43, 44, 15, 16, 17, 47, 50, 53, 46, 49, 52, 45, 48, 51]],
    [[0, 1, 2, 29, 32, 35, 51, 52, 53, 9, 12, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44] , [29, 32, 35, 53, 52, 51, 9, 12, 15, 2, 1, 0, 42, 39, 36, 43, 40, 37, 44, 41, 38]],
    [[2, 5, 8, 20, 23, 26, 47, 50, 53, 36, 39, 42, 27, 28, 29, 30, 31, 32, 33, 34, 35] , [20, 23, 26, 47, 50, 53, 42, 39, 36, 8, 5, 2, 33, 30, 27, 34, 31, 28, 35, 32, 29]],
    [[9, 10, 11, 36, 37, 38, 27, 28, 29, 18, 19, 20] + list(range(9)) , [36, 37, 38, 27, 28, 29, 18, 19, 20, 9, 10, 11, 2, 5, 8, 1, 4, 7, 0, 3, 6]],
    [[0, 3, 6, 18, 21, 24, 45, 48, 51, 38, 41, 44] + list(range(9, 18)), [18, 21, 24, 45, 48, 51, 44, 41, 38, 6, 3, 0, 11, 14, 17, 10, 13, 16, 9, 12, 15]],
    [[6, 7, 8, 27, 30, 33, 45, 46, 47, 11, 14, 17] + list(range(18, 27)) , [27, 30, 33, 47, 46, 45, 11, 14, 17, 8, 7, 6, 20, 23, 26, 19, 22, 25, 18, 21, 24]],
    [[2, 5, 8, 36, 39, 42, 47, 50, 53, 20, 23, 26] + list(range(27, 36)), [42, 39, 36, 53, 50, 47, 20, 23, 26, 2, 5, 8, 29, 32, 35, 28, 31, 34, 27, 30, 33]],
    [[6, 7, 8, 11, 14, 17, 45, 46, 47, 27, 30, 33] + list(range(18, 27)) , [17, 14, 11, 45, 46, 47, 33, 30, 27, 6, 7, 8, 24, 21, 18, 25, 22, 19, 26, 23, 20]],
    [[0, 1, 2, 9, 12, 15, 51, 52, 53, 29, 32, 35] + list(range(36, 45)), [15, 12, 9, 51, 52, 53, 35, 32, 29, 0, 1, 2, 38, 41, 44, 37, 40, 43, 36, 39, 42]]]
    
    assert len(perms) == len(solved_states) == len(states)
    fig = None
    action_to_move_lookup = {
    0: "D", 1: "U", 2: "L", 3: "D'", 4: "B", 5: "R",
    6: "U'", 7: "L'", 8: "F'", 9: "R'", 10: "F", 11: "B'"
    }

    for i, (perm, solved_state, state) in enumerate(zip(perms, solved_states, states)):
        solved_state = np.array(list(solved_state))
        solved_state[perm[0]] = solved_state[perm[1]]
        print(i)
        c = Cube(3, whiteplastic=False,  stickers=solved_state)
        fig = c.render(fig, flat=False).savefig("err.png", dpi=865 / c.N)
        plt.draw()
        plt.show()

        c = Cube(3, whiteplastic=False)
        
        move = action_to_move_lookup[i]
        if move[-1] == "'":
            c.move(move[0], 0, -1)
        else:
            c.move(move, 0 ,1)
        fig = c.render(fig, flat=False).savefig("err.png", dpi=865 / c.N)
        plt.draw()
        plt.show()

        
        
        assert (solved_state == np.array(list(state))).all(), f"{i}: \n {solved_state}, \n {state}, \n {np.where(solved_state != np.array(list(state)))}"

 
    # fig=None
    # for k, v in action_to_move_lookup.items():
    #     c = Cube(3, whiteplastic=False)

    #     c.move(v, 0, 1)
    #     c.render(fig, flat=True).savefig(f"test{k}.png", dpi=865 / c.N)

        
    # fig = None
    # for m, s in enumerate(states):
    #     c = Cube(3, whiteplastic=False,  stickers=s)
    #     fig = c.render(fig, flat=False).savefig("test%02d.png" % m, dpi=865 / c.N)
    #     plt.draw()
    #     import pdb; pdb.set_trace()
    #     # plt.pause
    
    # fig = None
    
    sss = ['byybyybwbrrowbwgyrwbrrrggoyygorgboggbbwooorrwwggowwoyy', 
 'yybyywbbbbbwwbwgyrrrorrggoywbrrgboggygoooorrwwggowwoyy']
    
    asss = sss[0]
    bsss = sss[1]
    import copy
    perm = perms[6]
    asss = np.array(list(asss))
    csss = copy.deepcopy(asss)
    csss[perm[0]] = csss[perm[1]]
    sss.append(csss)
    bsss = np.array(list(bsss))
    inds = np.where(csss != bsss)
    print(inds, csss[inds])
    print(inds, bsss[inds])
    # assert (csss == np.array(list(bsss))).all(), f"{i}: \n {csss}, \n {bsss}, \n {np.where(csss != np.array(list(bsss)))}"
    
    for m, s in enumerate(sss):
        c = Cube(3, whiteplastic=False,  stickers=s)
        fig = c.render(fig, flat=False).savefig("sss%02d.png" % m, dpi=865 / c.N)
        plt.draw()
        # plt.pause(0.001)
        # c.move("U", 0, 1)
        # print(c.get_state())

    