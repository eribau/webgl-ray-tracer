---
layout: layouts/blog.html
title: More spheres
tags: ["blog", "webgl", "Ray tracing"]
date: "2022-05-29"
---

I added some basic support in the shader for more than one sphere. Took me a while to figure out that you have to add an "out" qualifier to function's input parameters that are to be changed, similar to pass-by-reference in c++.

![Now with more ground](../images/ground.png "Now with more ground.")
