#!/usr/bin/env wolframscript
(* ::Package:: *)

xp = {a, e, f} |-> (a (1 - e^2)/(1 + e Cos[f])) Cos[f];
yp= {a, e, f} |-> (a (1 - e^2)/(1 + e Cos[f])) Sin[f];
zp= {a, e, f} |-> 0;
rotx = {v, \[Phi]} |->{{1,0,0},{0, Cos[\[Phi]], -Sin[\[Phi]]}, {0, Sin[\[Phi]], Cos[\[Phi]]}} . v;
rotz = {v, \[Phi]} |->{{Cos[\[Phi]], -Sin[\[Phi]], 0},{Sin[\[Phi]], Cos[\[Phi]], 0},{0,0,1}} . v;
roty = {v, \[Phi]} |-> {{Cos[\[Phi]], 0, Sin[\[Phi]]},{0,1,0},{-Sin[\[Phi]],0, Cos[\[Phi]]}} . v;

orb2SkyTransform = {v, \[Omega], i, \[CapitalOmega]} |->rotz[rotx[rotz[v, \[Omega]], i], \[CapitalOmega]];
sky2OrbTransform = {v, \[Omega], i, \[CapitalOmega]} |-> rotz[rotx[rotz[v, -\[CapitalOmega]], -i], -\[Omega]];

orb2PlanetTransform = {v, \[Phi], \[Theta]} |-> roty[rotz[v, -\[Theta]], -\[Phi]];
planet2OrbTransform = {v, \[Phi], \[Theta]} |-> rotz[roty[v, \[Phi]], \[Theta]];

skypos = {a, e, f, \[Omega], i, \[CapitalOmega]}|->Simplify[orb2SkyTransform[{xp[a, e,f], yp[a, e,f], zp[a, e,f]}, \[Omega], i, \[CapitalOmega]]];

(* this is x,y,z in the planet's frame as functions of x,y,z in the sky frame *)
w = Simplify[orb2PlanetTransform[sky2OrbTransform[{x, y, z}, \[Omega], i, \[CapitalOmega]]-{xp[a,e,f], yp[a,e,f], zp[a, e, f]}, \[Phi], \[Theta]]];

(* this is x,y,z in the sky frame as functions of x,y,z in planet frame *)
q = Simplify[orb2SkyTransform[planet2OrbTransform[{x,y,z},\[Phi],\[Theta]]+{xp[a,e,f],yp[a,e,f],zp[a,e,f]}, \[Omega],i,\[CapitalOmega]]];

files = {
StringJoin[NotebookDirectory[], "sky2plan-x_x.txt"],
StringJoin[NotebookDirectory[], "sky2plan-x_y.txt"],
StringJoin[NotebookDirectory[], "sky2plan-x_z.txt"],
StringJoin[NotebookDirectory[], "sky2plan-x_0.txt"]
};
coeffs = CoefficientRules[w[[1]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
 
files = {
StringJoin[NotebookDirectory[], "sky2plan-y_x.txt"],
StringJoin[NotebookDirectory[], "sky2plan-y_y.txt"],
StringJoin[NotebookDirectory[], "sky2plan-y_z.txt"],
StringJoin[NotebookDirectory[], "sky2plan-y_0.txt"]
};
coeffs = CoefficientRules[w[[2]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
 
 
files = {
StringJoin[NotebookDirectory[], "sky2plan-z_x.txt"],
StringJoin[NotebookDirectory[], "sky2plan-z_y.txt"],
StringJoin[NotebookDirectory[], "sky2plan-z_z.txt"],
StringJoin[NotebookDirectory[], "sky2plan-z_0.txt"]
};
coeffs = CoefficientRules[w[[3]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]


files = {
StringJoin[NotebookDirectory[], "plan2sky-x_x.txt"],
StringJoin[NotebookDirectory[], "plan2sky-x_y.txt"],
StringJoin[NotebookDirectory[], "plan2sky-x_z.txt"],
StringJoin[NotebookDirectory[], "plan2sky-x_0.txt"]
};
coeffs = CoefficientRules[q[[1]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
 
files = {
StringJoin[NotebookDirectory[], "plan2sky-y_x.txt"],
StringJoin[NotebookDirectory[], "plan2sky-y_y.txt"],
StringJoin[NotebookDirectory[], "plan2sky-y_z.txt"],
StringJoin[NotebookDirectory[], "plan2sky-y_0.txt"]
};
coeffs = CoefficientRules[q[[2]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
 
 
files = {
StringJoin[NotebookDirectory[], "plan2sky-z_x.txt"],
StringJoin[NotebookDirectory[], "plan2sky-z_y.txt"],
StringJoin[NotebookDirectory[], "plan2sky-z_z.txt"],
StringJoin[NotebookDirectory[], "plan2sky-z_0.txt"]
};
coeffs = CoefficientRules[q[[3]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
