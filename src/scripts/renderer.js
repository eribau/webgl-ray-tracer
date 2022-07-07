"use strict";

const glsl = (x) => x;

// Define shaders
//-------------------------------------------------------------------
// Vertex shader
const vertexShaderSource = glsl`#version 300 es
   in vec4 vertex;

   void main() {
      gl_Position = vertex;
   }
`;

// Fragment shader
const fragmentShaderHeader = glsl`#version 300 es
   precision mediump float;
   
   uniform float time;
   uniform float textureWeight;
   uniform sampler2D u_texture;
   out vec4 frag_color;

   // Image
   const float aspect_ratio = 16.0 / 9.0;
   const float image_width = 600.0;
   const float image_height = image_width / aspect_ratio;
   const float viewport_height = 2.0;
   const float viewport_width = viewport_height * aspect_ratio;
   const float focal_length = 1.0;
   const int samples_per_pixel = 1;
   const int max_depth = 50;

   // Camera
   const vec3 origin = vec3(0.0, 0.0, 0.0);
   const vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
   const vec3 vertical = vec3(0.0, viewport_height, 0.0);
   const vec3 lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - vec3(0.0, 0.0, focal_length);

   // Constants
   float infinity = 100000.0;
   float pi = 3.1415926535897932385;
   #define TAU 2. *pi
`;

// Utilities
const degreesToRadians = glsl`
   float degrees_to_radians(float degrees) {
      return degrees * pi / 180.0;
   }

   bool near_zero(vec3 vec) {
      float s = 1e-8;
      return (vec.x < s) && (vec.y < s) && (vec.z < s);
   }

   vec3 refraction(vec3 uv, vec3 n, float etai_over_etat) {
      float cos_theta = min(dot(-uv, n), 1.0);
      vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
      vec3 r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp)))*n;
      return r_out_perp + r_out_parallel;
   }

   float reflectance(float cosine, float ref_idx) {
      float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
      r0 = r0*r0;
      return r0 + (1.0 - r0)*pow((1.0 - cosine), 5.0);
   }
`;

const random = glsl`
   // Functions for generating pseudorandom numbers
   // Taken from https://www.shadertoy.com/view/llVcDz
   float g_seed = 0.25;

   uint base_hash(uvec2 p) {
      p = 1103515245U*((p >> 1U)^(p.yx));
      uint h32 = 1103515245U*((p.x)^(p.y>>3U));
      return h32^(h32 >> 16);
   }

   float hash1(inout float seed) {
      uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
      return float(n)*(1.0/float(0xffffffffU));
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
      vec3 h = hash3(seed) * vec3(2.,TAU,1.)-vec3(1,0,0);
      float phi = h.y;
      float r = pow(h.z, 1./3.);
      return r * vec3(sqrt(1.-h.x*h.x)*vec2(sin(phi),cos(phi)),h.x);
   }

   vec3 random_in_hemisphere(in vec3 normal) {
      vec3 in_unit_sphere = random_in_unit_sphere(g_seed);
      if(dot(in_unit_sphere, normal) > 0.0) {
         return in_unit_sphere;
      } else {
         return -in_unit_sphere;
      }
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

const camera = glsl`
   struct Camera {
      vec3 origin;
      vec3 horizontal;
      vec3 vertical;
      vec3 lower_left_corner;
   };

   Ray get_ray(Camera c, float u, float v) {
      return Ray(c.origin, normalize(c.lower_left_corner + u*c.horizontal + v*c.vertical - c.origin));
   }
`;

const material = glsl`
   struct Material {
      int material; // 0 = lambertian, 1 = metal, 2 = dielectric
      vec3 albedo;
      float fuzz; 
      float ir; // index of reflection for dielectrics
   };
`;

const hit_record = glsl`
   struct Hit_record {
      vec3 p;
      vec3 normal;
      float t;
      bool front_face;
      Material material;
   };
`;

const setFaceNormal = glsl`
   void set_face_normal(out Hit_record rec, Ray r, vec3 outward_normal) {
      rec.front_face = dot(r.direction, outward_normal) < 0.0;
      rec.normal = rec.front_face ? outward_normal: -outward_normal;
   }
`;

const scatter = glsl`
   bool lambertian_scatter(in Hit_record rec, inout vec3 attenuation, inout Ray scattered) {
      vec3 scatter_direction = random_in_hemisphere(rec.normal);

      if(near_zero(scatter_direction)) {
         scatter_direction = rec.normal;
      }
      scattered = Ray(rec.p, scatter_direction);
      attenuation = rec.material.albedo;
      return true;
   }

   bool metal_scatter(in Ray r_in, in Hit_record rec, inout vec3 attenuation, inout Ray scattered) {
      vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
      scattered = Ray(rec.p, reflected + rec.material.fuzz*random_in_unit_sphere(g_seed));
      attenuation = rec.material.albedo;
      return (dot(scattered.direction, rec.normal) > 0.0);
   }

   bool dielectric_scatter(in Ray r_in, in Hit_record rec, inout vec3 attenuation, inout Ray scattered) {
      attenuation = vec3(1.0);
      float refraction_ratio = rec.front_face ? (1.0/rec.material.ir) : rec.material.ir;

      vec3 unit_direction = normalize(r_in.direction);
      float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
      float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

      bool cannot_refract = refraction_ratio * sin_theta > 1.0;
      vec3 direction;

      if(cannot_refract || reflectance(cos_theta, refraction_ratio) > hash1(g_seed)) {
         direction = reflect(unit_direction, rec.normal);
      } else {
         direction = refract(unit_direction, rec.normal, refraction_ratio);
      }

      scattered = Ray(rec.p, direction);
      return true;
   }
`;

const sphere = glsl`
   struct Sphere {
      vec3 center;
      float radius;
      Material material;
   };
`;

const sphereHit = glsl`
   bool hit_sphere(const in Sphere sphere, 
                  const in Ray r,
                  const in float t_min,
                  const in float t_max,
                  inout Hit_record rec
   ) {
      vec3 oc = r.origin - sphere.center;
      float a = dot(r.direction, r.direction);
      float half_b = dot(oc, r.direction);
      float c = dot(oc, oc) - sphere.radius*sphere.radius;
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
      vec3 outward_normal = (rec.p - sphere.center) / sphere.radius;
      set_face_normal(rec, r, outward_normal);
      rec.material = sphere.material; 

      return true;
   }
`;

const hit = glsl`
   bool hit(Sphere spheres[4], Ray r, float t_min, float t_max, out Hit_record rec) {
      Hit_record temp_rec;
      bool hit_anything = false;
      float closest_so_far = t_max;

      for(int i = 0; i < 4; i++) {
         // TODO Rewrite hit_sphere to use Sphere instead of center and radius 
         if(hit_sphere(spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
         }
      }

      return hit_anything;
   }
`;

const rayColor = glsl`
   vec3 ray_color(Ray r, Sphere spheres[4]) {
      Hit_record rec;
      vec3 color = vec3(1.0);

      for(int i = 0; i < max_depth; i++) {
         if(hit(spheres, r, 0.001, infinity, rec)) {
            // vec3 target = rec.normal + normalize(random_in_unit_sphere(g_seed));
            // vec3 target = random_in_hemisphere(rec.normal);
            // color *= 0.5;

            // r.origin = rec.p;
            // r.direction = target;
            Ray scattered;
            vec3 attenuation;
            
            switch (rec.material.material) {
               case 0:
                  if(lambertian_scatter(rec, attenuation, scattered)) {
                     color *= attenuation;
                     r = scattered;
                  }
                  break;
               case 1:
                  if(metal_scatter(r, rec, attenuation, scattered)) {
                     color *= attenuation;
                     r = scattered;
                  }
                  break;
               case 2:
                  if(dielectric_scatter(r, rec, attenuation, scattered)) {
                     color *= attenuation;
                     r = scattered;
                  }
                  break;
            }
         } else {
            vec3 unit_direction = normalize(r.direction);
            float t = 0.5*(unit_direction.y + 1.0);
            color *= mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
            return color;
         }
      }
      return color;
   }
`;

const fragmentShaderMain = glsl`
   void main() {
      vec2 resolution = vec2(600, 338);
      float aspect = resolution.x / resolution.y;

      // Materials
      Material material_ground = Material(0, vec3(0.8, 0.8, 0.0), 0.0, 0.0);
      Material material_center = Material(0, vec3(0.1, 0.2, 0.5), 0.0, 0.0);
      Material material_left = Material(2, vec3(0.8, 0.8, 0.8), 0.3, 1.5);
      Material material_right = Material(1, vec3(0.8, 0.6, 0.2), 0.0, 0.0);

      // World
      Sphere spheres[4];
      spheres[0] = Sphere(vec3(0.0, 0.0, -1.0), 0.5, material_center);
      spheres[1] = Sphere(vec3(0.0, -100.5, -1.0), 100.0, material_ground);
      spheres[2] = Sphere(vec3(-1.0, 0.0, -1.0), 0.5, material_left);
      spheres[3] = Sphere(vec3(1.0, 0.0, -1.0), 0.5, material_right);

      // Set random generator seed
      g_seed = float(base_hash(floatBitsToUint(gl_FragCoord.xy)))/float(0xffffffffU)+time;

      // Get the coordinates to send the ray through
      vec2 uv = (gl_FragCoord.xy + hash2(g_seed)) / resolution;

      // Setup the camera
      Camera c = Camera(origin, horizontal, vertical, lower_left_corner);

      // Get the ray and the calculate the color for that "pixel"
      Ray r = get_ray(c, uv.x, uv.y);
      vec3 color = clamp(ray_color(r, spheres), 0.0, 0.999);
      
      vec3 texture = texture(u_texture, gl_FragCoord.xy / resolution).rgb;
      // vec3 texture = vec3(0.0);
      frag_color = vec4(mix(sqrt(color), texture, textureWeight), 1.0);
   }
`;

function createFragmentShaderSource() {
   return (
      fragmentShaderHeader +
      degreesToRadians +
      random +
      ray +
      rayAt +
      camera +
      material +
      hit_record +
      setFaceNormal +
      scatter +
      sphere +
      sphereHit +
      hit +
      rayColor +
      fragmentShaderMain
   );
}

var textureVertexSource = glsl`#version 300 es
   in vec4 vertex;
   in vec2 a_texcoord;

   out vec2 v_texcoord;

   void main() {
      gl_Position = vertex;

      v_texcoord = a_texcoord;
   }
`;

var textureFragmentSource = glsl`#version 300 es
   precision highp float;

   in vec2 v_texcoord;

   uniform sampler2D u_texture;

   out vec4 fragColor;

   void main() {
      vec3 texture = texture(u_texture, v_texcoord).rgb;
      fragColor = vec4(mix(texture, vec3(1.0, 0.0, 0.0), 0.25), 1.0);
   }
 `;

var renderVertexSource = glsl`#version 300 es
   in vec4 vertex;

   out vec2 v_texcoord;

   void main() {
      v_texcoord = vertex.xy * 0.5 + 0.5;
      gl_Position = vertex;
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

function tick(gl, timeSinceStart) {
   var textureProgram = createProgramFromSource(
      gl,
      vertexShaderSource,
      // reinFragmentSource
      createFragmentShaderSource()
   );

   // Setup texture
   var textureVertexAttribute = gl.getAttribLocation(textureProgram, "vertex");

   var timeLocation = gl.getUniformLocation(textureProgram, "time");
   var textureWeightLocation = gl.getUniformLocation(
      textureProgram,
      "textureWeight"
   );

   // Create a buffer for vertices
   var positionBuffer = gl.createBuffer();

   var textureVao = gl.createVertexArray();
   gl.bindVertexArray(textureVao);
   gl.enableVertexAttribArray(textureVertexAttribute);

   // Bind it to ARRAY_BUFFER
   gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

   // Set geometry (i.e. two triangles forming a rectangle)
   var positions = new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]);
   gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
   gl.vertexAttribPointer(textureVertexAttribute, 2, gl.FLOAT, false, 0, 0);

   // Create texture to render to
   var textures = [];
   gl.activeTexture(gl.TEXTURE0 + 0);
   for (var i = 0; i < 2; i++) {
      textures.push(gl.createTexture());
      gl.bindTexture(gl.TEXTURE_2D, textures[i]);
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
      // Skip mipmap
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
   }
   gl.bindTexture(gl.TEXTURE_2D, null);

   // Create and bind the framebuffer
   const framebuffer = gl.createFramebuffer();
   gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

   // Render to canvas
   var renderProgram = createProgramFromSource(
      gl,
      renderVertexSource,
      renderFragmentSource
   );

   var renderVertexAttribute = gl.getAttribLocation(renderProgram, "vertex");

   // Create a buffer for vertices
   var vertexBuffer = gl.createBuffer();

   var vao = gl.createVertexArray();
   gl.bindVertexArray(vao);
   gl.enableVertexAttribArray(renderVertexAttribute);

   // Bind it to ARRAY_BUFFER
   gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);

   // Set geometry (i.e. two triangles forming a rectangle)
   var vertices = new Float32Array([-1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1]);
   gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
   gl.vertexAttribPointer(renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);

   var sampleCount = 0;
   for (var i = 0; i < 100; i++) {
      // Render to the texture
      gl.bindTexture(gl.TEXTURE_2D, textures[0]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      // Attach the texture as the first color attachment
      gl.framebufferTexture2D(
         gl.FRAMEBUFFER,
         gl.COLOR_ATTACHMENT0,
         gl.TEXTURE_2D,
         textures[1],
         0
      );

      // Use textureProgram
      gl.useProgram(textureProgram);
      gl.bindVertexArray(textureVao);

      // set uniforms
      gl.uniform1f(timeLocation, timeSinceStart + i);
      gl.uniform1f(textureWeightLocation, sampleCount / (sampleCount + 1));
      // Draw texture and unbind framebuffer
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      // Ping pong textures
      textures.reverse();

      // Render to the canvas
      gl.useProgram(renderProgram);
      gl.bindVertexArray(vao);
      gl.bindTexture(gl.TEXTURE_2D, textures[0]);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      sampleCount += 1;
   }

   // requestAnimationFrame(tick(gl, timeSinceStart*0.001))
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

   const start = new Date();

   // Set canvas size
   resizeCanvasToDisplaySize(gl.canvas);

   // Set viewport
   gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

   var timeSinceStart = new Date() - start;
   // requestAnimationFrame(tick(gl, timeSinceStart));
   // setInterval(() => tick(gl, timeSinceStart), 1000 / 60);
   tick(gl, timeSinceStart);
}

window.onload = main;
