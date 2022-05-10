---
layout: layouts/blog.html
title: Hello world
tags: ['blog', 'webgl', 'Ray tracing']
date: '2022-05-10'
---

I started working on rendering an image in webGL today and got this "hello world" example working.

![Rainbow colored image](../images/hello_world.png "\"Hello world!\" Graphics edition")

So far I've been learning how webGL works from the site [WebGL Fundamentals](https://webglfundamentals.org/webgl/lessons/webgl-fundamentals.html) and I've adapted the code from the examples there to get this image from section 2.2 in [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#outputanimage). It's really nice that webGL handles all the interpolations and that I "only" have to hand it the positions and color data.

I also found a nice library from the [Khronos Group for debugging webGL](https://github.com/KhronosGroup/WebGLDeveloperTools) which let's me output all the function calls by webGL and any errors that might occur.