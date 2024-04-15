#!/usr/bin/env wolframscript
(* ::Package:: *)

rotx = {v, \[Phi]} |->{{1,0,0},{0, Cos[\[Phi]], -Sin[\[Phi]]}, {0, Sin[\[Phi]], Cos[\[Phi]]}} . v;
rotz = {v, \[Phi]} |->{{Cos[\[Phi]], -Sin[\[Phi]], 0},{Sin[\[Phi]], Cos[\[Phi]], 0},{0,0,1}} . v;
roty = {v, \[Phi]} |-> {{Cos[\[Phi]], 0, Sin[\[Phi]]},{0,1,0},{-Sin[\[Phi]],0, Cos[\[Phi]]}} . v;

xp = {a, e, f} |-> (a (1 - e^2)/(1 + e Cos[f])) Cos[f];
yp= {a, e, f} |-> (a (1 - e^2)/(1 + e Cos[f])) Sin[f];
zp= {a, e, f} |-> 0;

sky2OrbTransform = {v, \[Omega], i, \[CapitalOmega]} |-> rotz[rotx[rotz[v, -\[CapitalOmega]], -i], -\[Omega]];
orb2PlanetTransform = {v, \[Phi], \[Theta]} |-> roty[rotz[v, -\[Theta]], -\[Phi]];

sky2Planet= Simplify[orb2PlanetTransform[sky2OrbTransform[{x, y, z}, \[Omega], i, \[CapitalOmega]]-{xp[a,e,f], yp[a,e,f], zp[a, e, f]}, \[Phi], \[Theta]]];

q = Simplify[CoefficientRules[sky2Planet[[1]]^2/r^2, {x,y,z}]];
u = Simplify[CoefficientRules[sky2Planet[[2]]^2/(r^2 (1-f2)^2), {x,y,z}]];
o = Simplify[CoefficientRules[sky2Planet[[3]]^2/(r^2 (1-f1)^2), {x,y,z}]];

files = {
"planet_coeff_pxx.txt",
"planet_coeff_pxy.txt",
"planet_coeff_pxz.txt",
"planet_coeff_px0.txt",
"planet_coeff_pyy.txt",
"planet_coeff_pyz.txt",
"planet_coeff_py0.txt",
"planet_coeff_pzz.txt",
"planet_coeff_pz0.txt",
"planet_coeff_p00.txt"
}
Do[
Export[files[[v]],
 FortranForm[Simplify[q[[v]][[2]] +u[[v]][[2]] + o[[v]][[2]]]]], {v, 1, 10}]
