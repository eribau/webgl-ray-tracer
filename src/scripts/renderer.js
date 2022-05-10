"use strict";

const vertexShaderSource = 
'attribute vec4 a_position;' +
'attribute vec4 a_color;' +
'varying vec4 v_color;' +
'void main() {' +
'  gl_Position = a_position;' +
'  v_color = a_color;' +
'}';

const fragmentShaderSource = 
'precision mediump float;' +
'varying vec4 v_color;' +
'void main() {' +
'  gl_FragColor = v_color;' +
'}';

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

function createProgram(gl, vertexShader, fragmentShader) {
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

// function logGLCall(functionName, args) {   
//   console.log("gl." + functionName + "(" + 
//      WebGLDebugUtils.glFunctionArgsToString(functionName, args) + ")");   
// }

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

   var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
   var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

   var program = createProgram(gl, vertexShader, fragmentShader);

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

   var colorAttributeLocation = gl.getAttribLocation(program, "a_color");
   var colorBuffer = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
   
   var colors = [
      0, 0, 0.25, 1,
      0, 1, 0.25, 1,
      1, 0, 0.25, 1,
      1, 0, 0.25, 1,
      0, 1, 0.25, 1,
      1, 1, 0.25, 1,
   ];
   gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

   resizeCanvasToDisplaySize(gl.canvas);

   gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

   // Clear the canvas
   gl.clearColor(0, 0, 0, 0);
   gl.clear(gl.COLOR_BUFFER_BIT);

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

   gl.enableVertexAttribArray(colorAttributeLocation);

   gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);

   var size = 4;
   var type = gl.FLOAT;
   var normalize = false;
   var stride = 0;
   var offset = 0;
   gl.vertexAttribPointer(
      colorAttributeLocation, size, type, normalize, stride, offset);

   var primitiveType = gl.TRIANGLES;
   var offset = 0;
   var count = 6;
   gl.drawArrays(primitiveType, offset, count);
}

window.onload = main;