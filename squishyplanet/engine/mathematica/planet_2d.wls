#!/usr/bin/env wolframscript
(* ::Package:: *)

planet = pxx x^2 + pxy x y + pxz x z + px0 x + pyy y^2 + pyz y z + + py0 y + pzz z^2 + pz0 z + p00;
zsol = Solve[D[planet, z] == 0, z][[1]];
outline = Simplify[planet /. zsol];

q=CoefficientRules[outline, {x,y}]

files = {
"outline_coeff_exx.txt",
"outline_coeff_exy.txt",
"outline_coeff_ex0.txt",
"outline_coeff_eyy.txt",
"outline_coeff_ey0.txt",
"outline_coeff_e00.txt"}

Do[
Export[files[[v]],
 FortranForm[Simplify[q[[v]][[2]]]]], {v,1,6}]
