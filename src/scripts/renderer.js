"use strict";

const glsl = x => x;

// Define shaders
//-------------------------------------------------------------------
// Vertex shader
const vertexShaderSource = glsl`
   attribute vec4 a_position;
   void main() {
     gl_Position = a_position;
   }
`; 

// Fragment shader
const fragmentShaderHeader = glsl`
   precision mediump float;
   varying vec4 v_color;

   // Image
   float aspect_ratio = 16.0 / 9.0;
   float image_width = 600.0;
   float image_height = image_width / aspect_ratio;
   float viewport_height = 2.0;
   float viewport_width = viewport_height * aspect_ratio;
   float focal_length = 1.0;
   const int samples_per_pixel = 100;
   const int max_depth = 50;

   // Camera
   vec3 eye = vec3(0.0, 0.0, 0.0);
   vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
   vec3 vertical = vec3(0.0, viewport_height, 0.0);
   vec3 lower_left_corner = eye - horizontal/2.0 - vertical/2.0 - vec3(0.0, 0.0, focal_length);

   // Constants
   float infinity = 10000.0;
   float pi = 3.1415926535897932385;
`;

// Utilities
const degreesToRadians = glsl`
   float degrees_to_radians(float degrees) {
      return degrees * pi / 180.0;
   }
`;


const random = glsl`
   // Pseudo-random function taken from https://thebookofshaders.com/10/
   // float rand(){
   //    return fract(sin(dot(vec2(gl_FragCoord.x / image_width, gl_FragCoord / image_height), vec2(12.9898,78.233))) * 43758.5453);
   // }

   // Hash without Sine, taken from https://www.shadertoy.com/view/4djSRW
   // MIT License...
   /* Copyright (c)2014 David Hoskins.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.*/
   float rand()
   {
      vec2 p = vec2(gl_FragCoord.x / image_width, gl_FragCoord / image_height);
   	vec3 p3  = fract(vec3(p.xyx) * .1031);
       p3 += dot(p3, p3.yzx + 33.33);
       return fract((p3.x + p3.y) * p3.z);
   }

   float rand_range(float min, float max) {
      return min + (max-min)*rand();
   }

   // Based on https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/ and
   // https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
   vec3 random_in_unit_sphere() {
      float u = rand();
      vec3 p = vec3(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));

      float mag = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
      float c = pow(abs(u), 1.0 / 3.0);

      return p * c / mag;
   }
`;

const ray = glsl`
   struct Ray {
      vec3 origin;
      vec3 direction;
   };
`;

const rayAt = glsl`
   vec3 at(Ray r, float t) {
      return r.origin + t*r.direction;
   }  
`;

const hit_record = glsl`
   struct Hit_record {
      vec3 p;
      vec3 normal;
      float t;
      bool front_face;
   };
`;

const setFaceNormal = glsl`
   void set_face_normal(out Hit_record rec, Ray r, vec3 outward_normal) {
      rec.front_face = dot(r.direction, outward_normal) < 0.0;
      rec.normal = rec.front_face ? outward_normal: -outward_normal;
   }
`;

const sphere = glsl`
   struct Sphere {
      vec3 center;
      float radius;
   };
`;

const sphereHit = glsl`
   bool hit_sphere(vec3 center,
                  float radius,
                  Ray r,
                  float t_min,
                  float t_max,
                  out Hit_record rec
   ) {
      vec3 oc = r.origin - center;
      float a = dot(r.direction, r.direction);
      float half_b = dot(oc, r.direction);
      float c = dot(oc, oc) - radius*radius;
      float discriminant = half_b*half_b - a*c;
      if(discriminant < 0.0) {
         return false;
      }
      float sqrtd = sqrt(discriminant);

      float root = (-half_b - sqrtd) / a;
      if(root < t_min || t_max < root) {
         root = (-half_b + sqrtd) / a;
         if(root < t_min || t_max < root) {
            return false;
         }
      }

      rec.t = root;
      rec.p = at(r, rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      set_face_normal(rec, r, outward_normal);
      // rec.front_face = true;
      // rec.normal = outward_normal;

      return true;
   }
`;

const hit = glsl`
   bool hit(Sphere spheres[2], Ray r, float t_min, float t_max, out Hit_record rec) {
      Hit_record temp_rec;
      bool hit_anything = false;
      float closest_so_far = t_max;

      for(int i = 0; i < 2; i++) {
         // TODO Rewrite hit_sphere to use Sphere instead of center and radius 
         if(hit_sphere(spheres[i].center, spheres[i].radius, r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
         }
      }

      return hit_anything;
   }
`;

const rayColor = glsl`
   vec3 ray_color(Ray r, Sphere spheres[2]) {
      Hit_record rec;
      vec3 color = vec3(0.0, 0.0, 0.0);

      for(int i = 0; i < max_depth; ++i) {
         if(hit(spheres, r, 0.0, infinity, rec)) {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere();
            return 0.5 * (rec.normal + vec3(1.0, 1.0, 1.0));
         }
      }
      
      vec3 unit_direction = normalize(r.direction);
      float t = 0.5*(unit_direction.y + 1.0);
      return mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
   }
`;

const fragmentShaderMain = glsl`
   // World
   
   void main() {
      // World
      Sphere spheres[2];
      Sphere sphere = Sphere(vec3(0.0, 0.0, -1.0), 0.5);
      spheres[0] = Sphere(vec3(0.0, 0.0, -1.0), 0.5);
      spheres[1] = Sphere(vec3(0.0, -100.5, -1.0), 100.0);

      float f_samples_per_pixel = float(samples_per_pixel);
      float scale = 1.0 / f_samples_per_pixel;
      vec3 color = vec3(0.0, 0.0, 0.0);
      for(int s = 0; s < samples_per_pixel; ++s) {
         float u = (gl_FragCoord.x + rand()) / (image_width - 1.0);
         float v = (gl_FragCoord.y + rand()) / (image_height - 1.0);
         Ray r = Ray(eye, lower_left_corner + u*horizontal + v*vertical - eye);
         
         color += clamp(scale*ray_color(r, spheres), 0.0, 0.999);
      }
      gl_FragColor = vec4(color, 1.0);
   }
`;

function creatFragmentShaderSource() {
   return fragmentShaderHeader +
   degreesToRadians +
   random +
   ray +
   rayAt +
   hit_record + 
   setFaceNormal +
   sphere +
   sphereHit +
   hit +
   rayColor + 
   fragmentShaderMain;
}  

// Object classes
//-------------------------------------------------------------------
class Hittable {

}


//-------------------------------------------------------------------
function createShader(gl, type, source) {
   var shader = gl.createShader(type);
   gl.shaderSource(shader, source);
   gl.compileShader(shader);
   var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
   if(success) {
      return shader;
   }

   console.log(gl.getShaderInfoLog(shader));
   gl.deleteShader(shader);
}

function createProgramFromSource(gl, vertexSource, fragmentSource) {
   var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
   var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
   var program = gl.createProgram();
   gl.attachShader(program, vertexShader);
   gl.attachShader(program, fragmentShader);
   gl.linkProgram(program);
   var success = gl.getProgramParameter(program, gl.LINK_STATUS);
   if(success) {
      return program;
   }

   console.log(gl.getProgramInfoLog(program));
   gl.deteletProgram(program);
}

function logGLCall(functionName, args) {   
  console.log("gl." + functionName + "(" + 
     WebGLDebugUtils.glFunctionArgsToString(functionName, args) + ")");   
}

// From "WebGL Fundamentals"
function resizeCanvasToDisplaySize(canvas, multiplier) {
   multiplier = multiplier || 1;
   const width  = canvas.clientWidth  * multiplier | 0;
   const height = canvas.clientHeight * multiplier | 0;
   if (canvas.width !== width ||  canvas.height !== height) {
     canvas.width  = width;
     canvas.height = height;
     return true;
   }
   return false;
 }

//-------------------------------------------------------------------

function main() {
   var canvas = document.querySelector("#canvas");
   var gl = WebGLDebugUtils.makeDebugContext(canvas.getContext("webgl"), undefined, logGLCall);
   // var gl = canvas.getContext("webgl");
   if(!gl) {
      console.log("No webGl!");
      return;
   }

   var fragmentShaderSource = creatFragmentShaderSource();

   var program = createProgramFromSource(gl, vertexShaderSource, fragmentShaderSource);

   var positionAttributeLocation = gl.getAttribLocation(program, "a_position");
   var positionBuffer = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   var positions = [
      -1, -1,
      -1, 1,
      1, -1,
      1, -1,
      -1, 1,
      1, 1
   ];
   gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

   resizeCanvasToDisplaySize(gl.canvas);

   gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

   // Clear the canvas
   gl.clearColor(0, 0, 0, 0);
   gl.clear(gl.COLOR_BUFFER_BIT);

   // Render graphics 'hello world'
   gl.useProgram(program);

   gl.enableVertexAttribArray(positionAttributeLocation);

   // Bind the position buffer
   gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   var size = 2;
   var type = gl.FLOAT;
   var normalize = false;
   var stride = 0;
   var offset = 0;
   gl.vertexAttribPointer(
      positionAttributeLocation, size, type, normalize, stride, offset);

   var primitiveType = gl.TRIANGLES;
   var offset = 0;
   var count = 6;
   gl.drawArrays(primitiveType, offset, count);
}

window.onload = main;