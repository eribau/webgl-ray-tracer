"use strict";

const glsl = (x) => x;

// Define shaders
//-------------------------------------------------------------------
// Vertex shader
const vertexShaderSource = glsl`#version 300 es
   in vec4 a_position;
   void main() {
     gl_Position = a_position;
   }
`;

// Fragment shader
const fragmentShaderHeader = glsl`#version 300 es
   precision mediump float;
   out vec4 frag_color;

   // Image
   const float aspect_ratio = 16.0 / 9.0;
   const float image_width = 600.0;
   const float image_height = image_width / aspect_ratio;
   const float viewport_height = 2.0;
   const float viewport_width = viewport_height * aspect_ratio;
   const float focal_length = 1.0;
   const int samples_per_pixel = 100;
   const int max_depth = 50;

   // Camera
   const vec3 eye = vec3(0.0, 0.0, 0.0);
   const vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
   const vec3 vertical = vec3(0.0, viewport_height, 0.0);
   const vec3 lower_left_corner = eye - horizontal/2.0 - vertical/2.0 - vec3(0.0, 0.0, focal_length);

   // Constants
   float infinity = 10000.0;
   float pi = 3.1415926535897932385;
   #define TAU 2. *pi
   // Φ = Golden Ratio
   #define PHI 1.61803398874989484820459
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
      vec2 p = vec2(gl_FragCoord.x, gl_FragCoord.y);
   	vec3 p3  = fract(vec3(p.xyx) * .1031);
       p3 += dot(p3, p3.yzx + 33.33);
       return fract((p3.x + p3.y) * p3.z);
   }

   float rand_range(float min, float max) {
      return min + (max-min)*rand();
   }

   // https://github.com/Pikachuxxxx/Raytracing-in-a-Weekend-GLSL/blob/master/raytracer/shaders/Chapter-8-DiffuseMaterialsPS.glsl
   // float Random () {
   //    float phi = 1.61803398874989484820459;
   //    vec2 p = vec2(gl_FragCoord.x, gl_FragCoord.y);
   //    return fract(tan(distance(p*phi, p)*0.25)*p.x);
   // }

   // Based on https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/ and
   // https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
   // vec3 random_in_unit_sphere() {
   //    float u = rand();
   //    vec3 p = vec3(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));

   //    float mag = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
   //    float c = pow(abs(u), 1.0 / 3.0);
   //    p /= mag;

   //    return p * c;
   // }

   float g_seed = 0.25;
   float random (vec2 st) {
      return fract(tan(distance(st*PHI, st)*g_seed)*st.x);
   }

   vec2 random2(float seed){
     return vec2(
       random(vec2(seed-1.23, (seed+3.1)* 3.2)),
       random(vec2(seed+12.678, seed - 5.8324))
       );
   }

   vec3 random3(float seed){
     return vec3(
       random(vec2(seed-0.678, seed-0.123)),
       random(vec2(seed-0.3, seed+0.56)),
       random(vec2(seed+0.1234, seed-0.523))
       );
   }

   vec3 RandomInUnitSphere() {
      vec2 tp = vec2(rand(), rand());
      float theta = tp.x * 2. * pi;
      float phi = tp.y * 2. * pi;
      vec3 p = vec3(sin(theta) * cos(phi), sin(theta)*sin(phi), cos(theta));
    
      return normalize(p);
    }

    vec3 random_unit(float seed){
      vec2 rand = random2(seed);
      float a = rand.x * TAU;
      float z = (2. * rand.y) - 1.;
      float r = sqrt(1. - z*z);
      return vec3(r*cos(a), r*sin(a), z);
   }

   uint base_hash(uvec2 p) {
      p = 1103515245U*((p >> 1U)^(p.yx));
      uint h32 = 1103515245U*((p.x)^(p.y>>3U));
      return h32^(h32 >> 16);
   }

   vec2 hash2(inout float seed) {
       uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
       uvec2 rz = uvec2(n, n*48271U);
       return vec2(rz.xy & uvec2(0x7fffffffU))/float(0x7fffffff);
   }

   vec3 hash3(inout float seed) {
       uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
       uvec3 rz = uvec3(n, n*16807U, n*48271U);
       return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
   }

   vec3 random_in_unit_sphere(inout float seed) {
      vec3 h = hash3(seed) * vec3(2.,6.28318530718,1.)-vec3(1,0,0);
      float phi = h.y;
      float r = pow(h.z, 1./3.);
      return r * vec3(sqrt(1.-h.x*h.x)*vec2(sin(phi),cos(phi)),h.x);
   }

    //https://stackoverflow.com/a/34276128
   // bool isnan(float x){
   //    return !(x > 0. || x < 0. || x == 0.);
   // }
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
                  inout Hit_record rec
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
      vec3 color = vec3(1.0);
      float m = 1.0;

      // for(int i = 0; i < max_depth; ++i) {
      //    if(hit(spheres, r, 0.0, infinity, rec)) {
      //       vec3 target = rec.p + rec.normal + random_in_unit_sphere;
      //       r = Ray(rec.p, target - rec.p);
      //       m *= 0.5;
      //    } else {
      //       vec3 unit_direction = normalize(r.direction);
      //       float t = 0.5*(unit_direction.y + 1.0);
      //       color = mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
      //       break;
      //    }
      // }
      // return color * m;

      for(int i = 0; i < max_depth; ++i) {
         if(hit(spheres, r, 0.0, infinity, rec)) {
            vec3 target = rec.p + rec.normal + normalize(random_in_unit_sphere(g_seed));
            color *= 0.5;

            r.origin = rec.p;
            r.direction = target - rec.p;
         } else {
            vec3 unit_direction = normalize(r.direction);
            float t = 0.5*(unit_direction.y + 1.0);
            color = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
            return color;
         }
      }
      return color;
   }
`;

const fragmentShaderMain = glsl`
   // World
   
   void main() {
      // World
      Sphere spheres[2];
      spheres[0] = Sphere(vec3(0.0, 0.0, -1.0), 0.5);
      spheres[1] = Sphere(vec3(0.0, -100.5, -1.0), 100.0);

      float f_samples_per_pixel = float(samples_per_pixel);
      float scale = 1.0 / f_samples_per_pixel;
      vec3 color = vec3(0.0, 0.0, 0.0);
      for(int s = 0; s < samples_per_pixel; ++s) {

         // g_seed = random(gl_FragCoord.xy * (mod(float(s+11), 100.)));
         // if(isnan(g_seed)){
         //   g_seed = 0.25;
         // }

         g_seed = float(base_hash(floatBitsToUint(gl_FragCoord.xy)))/float(0xffffffffU)+float(s);

         float u = (gl_FragCoord.x + rand()) / (image_width - 1.0);
         float v = (gl_FragCoord.y + rand()) / (image_height - 1.0);
         Ray r = Ray(eye, lower_left_corner + u*horizontal + v*vertical - eye);
         
         color += clamp(scale*ray_color(r, spheres), 0.0, 0.999);
      }
      frag_color = vec4(sqrt(color), 1.0);
   }
`;

function creatFragmentShaderSource() {
   return (
      fragmentShaderHeader +
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
      fragmentShaderMain
   );
}

var textureVertexSource = glsl`#version 300 es
   in vec4 a_position;
   in vec2 a_texcoord;

   out vec2 v_texcoord;

   void main() {
      gl_Position = a_position;

      v_texcoord = a_texcoord;
   }
`;

var textureFragmentSource = glsl`#version 300 es
   precision highp float;

   in vec2 v_texcoord;

   uniform sampler2D u_texture;

   out vec4 fragColor;

   void main() {
      // fragColor = texture(u_texture, v_texcoord);
      fragColor = vec4(1.0, 1.0, 0.0, 1.0);
   }
 `;

var renderVertexSource = glsl`#version 300 es
   in vec4 a_position;
   in vec2 a_texcoord;

   out vec2 v_texcoord;

   void main() {
      v_texcoord = a_texcoord;
      gl_Position = a_position;
   }
`;

var renderFragmentSource = glsl`#version 300 es
   precision highp float;

   in vec2 v_texcoord;

   uniform sampler2D u_texture;

   out vec4 fragColor;

   void main() {
      fragColor = texture(u_texture, v_texcoord);
   }
`;

// Object classes
//-------------------------------------------------------------------
class Hittable {}

//-------------------------------------------------------------------
function createShader(gl, type, source) {
   var shader = gl.createShader(type);
   gl.shaderSource(shader, source);
   gl.compileShader(shader);
   var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
   if (success) {
      return shader;
   }

   console.log(gl.getShaderInfoLog(shader));
   gl.deleteShader(shader);
}

function createProgramFromSource(gl, vertexSource, fragmentSource) {
   var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
   var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
   var textureProgram = gl.createProgram();
   gl.attachShader(textureProgram, vertexShader);
   gl.attachShader(textureProgram, fragmentShader);
   gl.linkProgram(textureProgram);
   var success = gl.getProgramParameter(textureProgram, gl.LINK_STATUS);
   if (success) {
      return textureProgram;
   }

   console.log(gl.getProgramInfoLog(textureProgram));
   gl.deteletProgram(textureProgram);
}

function logGLCall(functionName, args) {
   console.log(
      "gl." +
         functionName +
         "(" +
         WebGLDebugUtils.glFunctionArgsToString(functionName, args) +
         ")"
   );
}

// From "WebGL Fundamentals"
function resizeCanvasToDisplaySize(canvas, multiplier) {
   multiplier = multiplier || 1;
   const width = (canvas.clientWidth * multiplier) | 0;
   const height = (canvas.clientHeight * multiplier) | 0;
   if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      return true;
   }
   return false;
}

//-------------------------------------------------------------------

function main() {
   var canvas = document.querySelector("#canvas");
   var gl = WebGLDebugUtils.makeDebugContext(
      canvas.getContext("webgl2"),
      undefined,
      logGLCall
   );
   // var gl = canvas.getContext("webgl2");
   if (!gl) {
      console.log("No webGl!");
      return;
   }

   // var fragmentShaderSource = creatFragmentShaderSource();

   // var textureProgram = createProgramFromSource(gl, vertexShaderSource, fragmentShaderSource);

   // var positionAttributeLocation = gl.getAttribLocation(textureProgram, "a_position");
   // var positionBuffer = gl.createBuffer();
   // gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   // var positions = [
   //    -1, -1,
   //    -1, 1,
   //    1, -1,
   //    1, -1,
   //    -1, 1,
   //    1, 1
   // ];
   // gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

   // resizeCanvasToDisplaySize(gl.canvas);

   // gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

   // // Clear the canvas
   // gl.clearColor(0, 0, 0, 0);
   // gl.clear(gl.COLOR_BUFFER_BIT);

   // // Render graphics 'hello world'
   // gl.useProgram(textureProgram);

   // gl.enableVertexAttribArray(positionAttributeLocation);

   // // Bind the position buffer
   // gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   // var size = 2;
   // var type = gl.FLOAT;
   // var normalize = false;
   // var stride = 0;
   // var offset = 0;
   // gl.vertexAttribPointer(
   //    positionAttributeLocation, size, type, normalize, stride, offset);

   // var primitiveType = gl.TRIANGLES;
   // var offset = 0;
   // var count = 6;
   // gl.drawArrays(primitiveType, offset, count);

   var textureProgram = createProgramFromSource(
      gl,
      textureVertexSource,
      textureFragmentSource
   );

   var positionAttributeLocation = gl.getAttribLocation(
      textureProgram,
      "a_position"
   );
   var texcoordAttributeLocation = gl.getAttribLocation(
      textureProgram,
      "a_texcoord"
   );

   // Create a buffer for position a
   var positionBuffer = gl.createBuffer();

   var vao = gl.createVertexArray();
   gl.bindVertexArray(vao);
   gl.enableVertexAttribArray(positionAttributeLocation);

   // Bind it to ARRAY_BUFFER
   gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   // Set geometry (i.e. two triangles forming a rectangle)
   var positions = new Float32Array([-1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1]);
   gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
   gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

   // Create the texcoord buffer and set the ARRAY_BUFFER to it
   var texcoordBuffer = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
   gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
   gl.vertexAttribPointer(texcoordAttributeLocation, 2, gl.FLOAT, true, 0, 0);

   // Create texture to render to
   var texture = gl.createTexture();
   gl.activeTexture(gl.TEXTURE0 + 0);
   gl.bindTexture(gl.TEXTURE_2D, texture);
   gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.canvas.width,
      gl.canvas.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null
   );
   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

   // Create and bind the framebuffer
   const framebuffer = gl.createFramebuffer();
   gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

   // Attach the texture as the first color attachment
   gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
   );

   // Render to the texture
   resizeCanvasToDisplaySize(gl.canvas);

   gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
   gl.bindTexture(gl.TEXTURE_2D, texture);
   gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
   gl.clearColor(0, 0, 0, 0);
   gl.clear(gl.COLOR_BUFFER_BIT);

   // Use textureProgram
   gl.useProgram(textureProgram);
   gl.bindVertexArray(vao);

   gl.drawArrays(gl.TRIANGLES, 0, 6);

   // Render to canvas
   var renderProgram = createProgramFromSource(
      gl,
      renderVertexSource,
      renderFragmentSource
   );

   positionAttributeLocation = gl.getAttribLocation(
      renderProgram,
      "a_position"
   );
   texcoordAttributeLocation = gl.getAttribLocation(
      renderProgram,
      "a_texcoord"
   );

   gl.bindVertexArray(vao);
   gl.enableVertexAttribArray(positionAttributeLocation);

   // Bind it to ARRAY_BUFFER
   gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   // Set geometry (i.e. two triangles forming a rectangle)
   gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
   gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

   // Create the texcoord buffer and set the ARRAY_BUFFER to it
   gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
   gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
   gl.vertexAttribPointer(texcoordAttributeLocation, 2, gl.FLOAT, true, 0, 0);

   gl.bindFramebuffer(gl.FRAMEBUFFER, null);
   gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
   gl.clearColor(0, 0, 0, 0);

   gl.useProgram(renderProgram);
   gl.bindVertexArray(vao);
   gl.drawArrays(gl.TRIANGLES, 0, 6)

}

window.onload = main;
