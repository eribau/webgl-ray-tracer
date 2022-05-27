"use strict";

const glsl = x => x;

// Define shaders
const vertexShaderSource = glsl`
   attribute vec4 a_position;
   void main() {
     gl_Position = a_position;
   }
`; 

const fragmentShaderSource = glsl` 
   precision mediump float;
   varying vec4 v_color;
   float aspect_ratio = 16.0 / 9.0;
   float image_width = 600.0;
   float image_height = image_width / aspect_ratio;
   float viewport_height = 2.0;
   float viewport_width = viewport_height * aspect_ratio;
   float focal_length = 1.0;

   vec3 eye = vec3(0.0, 0.0, 0.0);
   vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
   vec3 vertical = vec3(0.0, viewport_height, 0.0);
   vec3 lower_left_corner = eye - horizontal/2.0 - vertical/2.0 - vec3(0.0, 0.0, focal_length);

   vec3 at(vec3 origin, vec3 direction, float t) {
      return origin + t*direction;
   }

   float hit_sphere(vec3 center, float radius, vec3 origin, vec3 direction) {
      vec3 oc = origin - center;
      float a = dot(direction, direction);
      float half_b = dot(oc, direction);
      float c = dot(oc, oc) - radius*radius;
      float discriminant = half_b*half_b - a*c;
      if(discriminant < 0.0) {
         return -1.0;
      } else {
         return (-half_b - sqrt(discriminant)) / a;
      }
   }  

   vec3 ray_color(vec3 origin, vec3 direction) {
      float t = hit_sphere(vec3(0.0, 0.0, -1.0), 0.5, origin, direction);
      if(t > 0.0) {
         vec3 N = normalize(at(origin, direction, t) - vec3(0.0, 0.0, -1.0));
         return 0.5*vec3(N.x + 1.0, N.y + 1.0, N.z + 1.0);
      }
      vec3 unit_direction = normalize(direction);
      t = 0.5*(unit_direction.y + 1.0);
      return mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
   }

   void main() {
     float u = gl_FragCoord.x / image_width;
     float v = gl_FragCoord.y / image_height;
     gl_FragColor = vec4(ray_color(eye, lower_left_corner + u*horizontal + v*vertical - eye), 1.0);
   }
`;

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

function main() {
   var canvas = document.querySelector("#canvas");
   // var gl = WebGLDebugUtils.makeDebugContext(canvas.getContext("webgl"), undefined, logGLCall);
   var gl = canvas.getContext("webgl");
   if(!gl) {
      console.log("No webGl!");
      return;
   }

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