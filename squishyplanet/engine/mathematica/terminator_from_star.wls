#!/usr/bin/env wolframscript
(* ::Package:: *)

(* this is the planet in the coordinate system centered on the star, observer at z= infinity *)
planet = pxx x^2 + pxy x y +pxz x z + px0 x + pyy y^2 + pyz y z + py0 y + pzz z^2 + pz0 z + p00;

(* these are the rotations needed to convert that system into one where the planet is
straight along the z axis *)
rz = {{xc/Sqrt[xc^2+yc^2],yc/Sqrt[xc^2+yc^2], 0},{-(yc/Sqrt[xc^2+yc^2]), xc/Sqrt[xc^2+yc^2],0},{0,0,1}};
ry = {{zc/Sqrt[xc^2+yc^2+zc^2],0, -(Sqrt[xc^2+yc^2]/Sqrt[xc^2+yc^2+zc^2])},{0,1,0},{Sqrt[xc^2+yc^2]/Sqrt[xc^2+yc^2+zc^2], 0, zc/Sqrt[xc^2+yc^2+zc^2]}};
rotMat = Simplify[(ry . rz)];
invRotMat = Simplify[Inverse[rotMat]];

(* this is x, y, z in the rotated system where the planet is straight up the zr axis *)
transform = invRotMat . {xr, yr, zr};

(* transform the planet: *)
planetFromStar = Simplify[pxx x^2 + pxy x y +pxz x z + px0 x + pyy y^2 + pyz y z + py0 y + pzz z^2 + pz0 z + p00 /.
{x-> transform[[1]], y-> transform[[2]], z-> transform[[3]]}];

(* this is still just a quadratic in the new xr, yr, zr variables *)
coeffs = Simplify[CoefficientRules[planetFromStar, {xr, yr, zr}]];

(* save them all *)
files = {
StringJoin[NotebookDirectory[], "/terminator_coeff_pxx.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_pxy.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_pxz.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_px0.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_pyy.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_pyz.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_py0.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_pzz.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_pz0.txt"],
StringJoin[NotebookDirectory[], "/terminator_coeff_p00.txt"]
};

Do[
Export[files[[v]],
 FortranForm[coeffs[[v]][[2]]]], {v, 1, 10}]
