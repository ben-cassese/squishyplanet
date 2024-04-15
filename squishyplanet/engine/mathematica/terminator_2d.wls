#!/usr/bin/env wolframscript
(* ::Package:: *)

planet = pxx x^2 + pxy x y + pxz x z + px0 x + pyy y^2 + pyz y z + + py0 y + pzz z^2 + pz0 z + p00;
coeffs = FullSimplify[CoefficientRules[planet /. Solve[Grad[planet, {x,y,z}] . {xc,yc,zc}  == 0, z][[1]], {x,y}]];

files = {
    "terminator_coeff_txx.txt",
    "terminator_coeff_txy.txt",
    "terminator_coeff_tx0.txt",
    "terminator_coeff_tyy.txt",
    "terminator_coeff_ty0.txt",
    "terminator_coeff_t00.txt"}

Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v,1,6}]

intersectionSols = Simplify[Solve[{planet==1, Grad[planet, {x,y,z}] . {xc,yc,zc}  == 0, D[planet, z] == 0}, {x,y,z}]];

files = {
    "terminator_intersection_x1.txt",
    "terminator_intersection_y1.txt",
    "terminator_intersection_z1.txt"}

Do[
Export[files[[v]],
 FortranForm[Simplify[intersectionSols[[1]][[v]][[2]]], {v,1,3}]]]

files = {
    "terminator_intersection_x2.txt",
    "terminator_intersection_y2.txt",
    "terminator_intersection_z2.txt"}

Do[
Export[files[[v]],
 FortranForm[Simplify[intersectionSols[[2]][[v]][[2]]], {v,1,3}]]]
