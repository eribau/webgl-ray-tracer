---
layout: layouts/blog.html
title: Living in the material world
tags: ["blog", "webgl", "Ray tracing"]
date: "2022-07-07"
---

I've added other materials than the lambertian reflection model to the code. The first one is a metal material that reflects the light of it. It has a variable for controlling fuzziness, which basically means that incoming rays are reflected back with some randomness added to their direction.

![Metallic spheres](../images/fuzzy_metal.png "Sphere with lambertian reflection at the center alongside two metal spheres displaying varying degrees of fuzziness.")

The other material is a dielectric one, i.e. incoming light is both reflected and transmitted through the material. I'm not quite sure however if I've managed to get it to work properly. The reflection in the sphere looks different from the example image in the book, or it looks different when I add the Schlick approximation. The only difference that I can think of right now however is the function for generating random numbers, but it's possible that there is something else that I'm missing.

![Dielectric spheres](../images/dielectrics.png "Sphere on the left with dielectric material.")
