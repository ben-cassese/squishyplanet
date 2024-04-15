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

files = {
StringJoin[NotebookDirectory[], "x_x.txt"],
StringJoin[NotebookDirectory[], "x_y.txt"],
StringJoin[NotebookDirectory[], "x_z.txt"],
StringJoin[NotebookDirectory[], "x_0.txt"]
};
coeffs = CoefficientRules[w[[1]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
 
files = {
StringJoin[NotebookDirectory[], "y_x.txt"],
StringJoin[NotebookDirectory[], "y_y.txt"],
StringJoin[NotebookDirectory[], "y_z.txt"],
StringJoin[NotebookDirectory[], "y_0.txt"]
};
coeffs = CoefficientRules[w[[2]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]
 
 
files = {
StringJoin[NotebookDirectory[], "z_x.txt"],
StringJoin[NotebookDirectory[], "z_y.txt"],
StringJoin[NotebookDirectory[], "z_z.txt"],
StringJoin[NotebookDirectory[], "z_0.txt"]
};
coeffs = CoefficientRules[w[[3]], {x,y,z}];
Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v, 1, 4}]


profile = Simplify[(\[Kappa]/(2 Pi (Exp[\[Kappa]] - Exp[-\[Kappa]])) Exp[\[Kappa] ({Sin[\[Alpha]]Cos[\[Beta]], Sin[\[Alpha]]Sin[\[Beta]], Cos[\[Alpha]]} . x)/Sqrt[x . x]] /. {x-> {x,y,z}}) /. {x-> x/r, y-> y/(r(1-f2)), z-> z/(r(1-f1))}];

Export[StringJoin[NotebookDirectory[], "profile.txt"], FortranForm[profile]];
