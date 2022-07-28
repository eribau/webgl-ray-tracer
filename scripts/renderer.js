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
const fragmentShaderHeader = (number_of_spheres) => {
   return `#version 300 es
      precision mediump float;

      uniform float time;
      uniform float textureWeight;
      uniform sampler2D u_texture;
      uniform vec2 res;
      out vec4 frag_color;

      // Image
      const int max_depth = 50;

      // Camera
      uniform vec3 eye;

      // Constants
      float infinity = 100000.0;
      float pi = 3.1415926535897932385;
      #define TAU 2. *pi
      #define NUMBER_OF_SPHERES ${number_of_spheres}
   `;
};

// Utilities
const utilities = glsl`
   float degrees_to_radians(float degrees) {
      return degrees * pi / 180.0;
   }

   bool near_zero(vec3 vec) {
      float s = 1e-8;
      return (vec.x < s) && (vec.y < s) && (vec.z < s);
   }

   vec3 reflection(in vec3 v, in vec3 n) {
      return v - 2.0*dot(v, n)*n;
   }

   vec3 refraction(vec3 uv, vec3 n, float etai_over_etat) {
      float cos_theta = min(dot(-uv, n), 1.0);
      vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
      vec3 r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp)))*n;
      return r_out_perp + r_out_parallel;
   }

   float reflectance(const in float cosine, const in float ref_idx) {
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

   vec2 random_in_unit_disk(inout float seed) {
      vec2 h = hash2(seed) * vec2(1.,6.28318530718);
      float phi = h.y;
      float r = sqrt(h.x);
     return r * vec2(sin(phi),cos(phi));
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
      float lens_radius;
      vec3 u, v, w;
   };

   Camera init_camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist) {
      float theta = degrees_to_radians(vfov);
      float h = tan(theta/2.0);
      float viewport_height = 2.0 * h;
      float viewport_width = aspect_ratio * viewport_height;

      vec3 w = normalize(lookfrom - lookat);
      vec3 u = normalize(cross(vup, w));
      vec3 v = cross(w, u);

      vec3 origin = lookfrom;
      vec3 horizontal = focus_dist * viewport_width * u;
      vec3 vertical = focus_dist * viewport_height * v;
      vec3 lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - focus_dist*w;

      float lens_radius = aperture / 2.0;

      return Camera(origin, horizontal, vertical, lower_left_corner, lens_radius, u, v, w);
   }

   Ray get_ray(Camera c, float s, float t) {
      vec2 rd = c.lens_radius * random_in_unit_disk(g_seed);
      vec3 offset = c.u * rd.x + c.v * rd.y;

      return Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal + t*c.vertical - c.origin - offset));
   }
`;

const material = glsl`
   #define LAMBERTIAN 0
   #define METAL 1
   #define DIELECTRIC 2

   struct Material {
      int type; // 0 = lambertian, 1 = metal, 2 = dielectric
      vec3 albedo; // The color of the material
      float fuzz; // Metallic fuzziness
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
      // vec3 scatter_direction = random_in_hemisphere(rec.normal);
      vec3 scatter_direction = normalize(rec.normal + random_in_unit_sphere(g_seed));

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

   bool scatter(in Ray r_in, in Hit_record rec, inout vec3 attenuation, inout Ray scattered) {
      switch (rec.material.type) {
         case LAMBERTIAN:
            return lambertian_scatter(rec, attenuation, scattered);
            break;
         case METAL:
            return metal_scatter(r_in, rec, attenuation, scattered);
            break;
         case DIELECTRIC:
            return dielectric_scatter(r_in, rec, attenuation, scattered);
            break;
         default:
            return false;
      }
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
      // rec.normal = (rec.p - sphere.center) / sphere.radius;
      rec.material = sphere.material;

      return true;
   }
`;

const hit = glsl`
   bool hit(const in Sphere spheres[NUMBER_OF_SPHERES], 
            const in Ray r, 
            const in float t_min, const in 
            float t_max, 
            out Hit_record rec
   ) {
      Hit_record temp_rec;
      bool hit_anything = false;
      float closest_so_far = t_max;

      for(int i = 0; i < NUMBER_OF_SPHERES; i++) {
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
   vec3 ray_color(in Ray r, in Sphere spheres[NUMBER_OF_SPHERES]) {
      Hit_record rec;
      vec3 color = vec3(1.0);

      for(int i = 0; i < max_depth; i++) {
         if(hit(spheres, r, 0.001, infinity, rec)) {
            Ray scattered;
            vec3 attenuation;

            if(scatter(r, rec, attenuation, scattered)) {
               color *= attenuation;
               r = scattered;
            } else {
               return vec3(0.0);
            }
         } else {
            vec3 unit_direction = normalize(r.direction);
            float t = 0.5*unit_direction.y + 0.5;
            color *= mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
            return color;
         }
      }
      return vec3(0.0);
   }
`;

const fragmentShaderMain = (world) => {
   return `
      void main() {
         float aspect = res.x / res.y;

         // World
         Sphere spheres[NUMBER_OF_SPHERES];

         ${world}

         // Set random generator seed
         g_seed = float(base_hash(floatBitsToUint(gl_FragCoord.xy)))/float(0xffffffffU)+time;

         // Get the coordinates to send the ray through
         vec2 uv = (gl_FragCoord.xy + hash2(g_seed)) / res;

         // Setup the camera
         // Camera c = Camera(origin, horizontal, vertical, lower_left_corner, lens_radius, u, v, w);
         vec3 lookfrom = eye;
         vec3 lookat = vec3(0.0, 0.0, 0.0);
         vec3 vup = vec3(0.0, 1.0, 0.0);
         float fov = 20.0;
         float dist_to_focus = length(lookfrom - lookat);
         // float dist_to_focus = 10.0;
         float aperture = 0.1;
         Camera c = init_camera(lookfrom, lookat, vup, fov, aspect, aperture, dist_to_focus);

         // Get the ray and the calculate the color for that "pixel"
         Ray r = get_ray(c, uv.x, uv.y);
         vec3 color = clamp(ray_color(r, spheres), 0.0, 0.999);

         vec3 texture = texture(u_texture, gl_FragCoord.xy / res).rgb;
         // vec3 texture = vec3(0.0);
         frag_color = vec4(mix(sqrt(color), texture, textureWeight), 1.0);
      }
   `;
};

function createFragmentShaderSource(number_of_spheres, world) {
   return (
      fragmentShaderHeader(number_of_spheres) +
      utilities +
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
      fragmentShaderMain(world)
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
const format = (number) => {
   return number.toLocaleString("en-us", {
      minimumFractionDigits: 1,
      useGrouping: false,
   });
};
class Material {
   type = 0;
   albedo = glMatrix.vec3.create();
   fuzz = 0;
   ir = 0;

   constructor(type, albedo, fuzz, ir) {
      this.type = type;
      this.albedo = albedo;
      this.fuzz = fuzz;
      this.ir = ir;
   }

   toString() {
      return `Material(${this.type}, vec3(${format(this.albedo[0])}, ${format(
         this.albedo[1]
      )}, ${format(this.albedo[2])}), ${format(this.fuzz)}, ${format(
         this.ir
      )})`;
   }
}
class Sphere {
   center = glMatrix.vec3.create();
   radius = 1;
   material = new Material(0, glMatrix.vec3.create(), 0, 0);

   constructor(center, radius, material) {
      this.center = center;
      this.radius = radius;
      this.material = material;
   }

   toString() {
      return `Sphere(vec3(${format(this.center[0])}, ${format(
         this.center[1]
      )}, ${format(this.center[2])}), ${format(
         this.radius
      )}, ${this.material.toString()})`;
   }
}

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

function degreesToRadians(degrees) {
   return (degrees * Math.PI) / 180.0;
}

function radiansToDegrees(radians) {
   return (radians * 180.0) / Math.PI;
}

class Renderer {
   gl;
   number_of_spheres;
   world_string;
   sampleCount;
   timeSinceStart;
   cameraPos;
   lookat;
   resolution;

   vertices;

   textureProgram;
   textureVertexAttribute;
   timeLocation;
   textureWeightLocation;
   eyeLocation;
   resolutionLocation;
   textureVertexBuffer;
   textureVao;
   textures;
   framebuffer;

   renderProgram;
   renderVertexAttribute;
   renderVertexBuffer;
   renderVao;

   constructor(gl, timeSinceStart, width, height) {
      this.gl = gl;
      this.vertices = new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]);
      this.sampleCount = 0;
      this.timeSinceStart = timeSinceStart;
      [this.number_of_spheres, this.world_string] = this.#setupWorld();
      this.cameraPos = glMatrix.vec3.fromValues(13, 2, 3);
      this.lookat = glMatrix.vec3.fromValues(0, 0, 0);
      this.resolution = glMatrix.vec2.fromValues(width, height);

      this.textureProgram = createProgramFromSource(
         this.gl,
         vertexShaderSource,
         createFragmentShaderSource(this.number_of_spheres, this.world_string)
      );

      // Setup texture
      this.textureVertexAttribute = this.gl.getAttribLocation(
         this.textureProgram,
         "vertex"
      );

      this.timeLocation = this.gl.getUniformLocation(
         this.textureProgram,
         "time"
      );
      this.textureWeightLocation = this.gl.getUniformLocation(
         this.textureProgram,
         "textureWeight"
      );
      this.resolutionLocation = this.gl.getUniformLocation(
         this.textureProgram,
         "res"
      );

      this.eyeLocation = this.gl.getUniformLocation(this.textureProgram, "eye");

      // Create a buffer for vertices
      this.textureVertexBuffer = this.gl.createBuffer();

      this.textureVao = this.gl.createVertexArray();
      this.gl.bindVertexArray(this.textureVao);
      this.gl.enableVertexAttribArray(this.textureVertexAttribute);

      // Bind it to ARRAY_BUFFER
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.textureVertexBuffer);

      // Set geometry (i.e. two triangles forming a rectangle)
      this.gl.bufferData(
         this.gl.ARRAY_BUFFER,
         this.vertices,
         this.gl.STATIC_DRAW
      );
      this.gl.vertexAttribPointer(
         this.textureVertexAttribute,
         2,
         this.gl.FLOAT,
         false,
         0,
         0
      );

      // Create textures to render to
      this.textures = [];
      this.gl.activeTexture(this.gl.TEXTURE0 + 0);
      for (var i = 0; i < 2; i++) {
         this.textures.push(this.gl.createTexture());
         this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures[i]);
         this.gl.texImage2D(
            this.gl.TEXTURE_2D,
            0,
            this.gl.RGBA,
            this.gl.canvas.width,
            this.gl.canvas.height,
            0,
            this.gl.RGBA,
            this.gl.UNSIGNED_BYTE,
            null
         );
         // Skip mipmap
         this.gl.texParameteri(
            this.gl.TEXTURE_2D,
            this.gl.TEXTURE_MIN_FILTER,
            this.gl.NEAREST
         );
         this.gl.texParameteri(
            this.gl.TEXTURE_2D,
            this.gl.TEXTURE_MAG_FILTER,
            this.gl.NEAREST
         );
         this.gl.texParameteri(
            this.gl.TEXTURE_2D,
            this.gl.TEXTURE_WRAP_S,
            this.gl.CLAMP_TO_EDGE
         );
         this.gl.texParameteri(
            this.gl.TEXTURE_2D,
            this.gl.TEXTURE_WRAP_T,
            this.gl.CLAMP_TO_EDGE
         );
      }
      this.gl.bindTexture(this.gl.TEXTURE_2D, null);

      // Create and bind the framebuffer
      this.framebuffer = this.gl.createFramebuffer();
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);

      // Setup program for rendering to canvas
      this.renderProgram = createProgramFromSource(
         this.gl,
         renderVertexSource,
         renderFragmentSource
      );

      this.renderVertexAttribute = this.gl.getAttribLocation(
         this.renderProgram,
         "vertex"
      );

      // Create a buffer for vertices
      this.renderVertexBuffer = this.gl.createBuffer();

      this.renderVao = this.gl.createVertexArray();
      this.gl.bindVertexArray(this.renderVao);
      this.gl.enableVertexAttribArray(this.renderVertexAttribute);

      // Bind it to ARRAY_BUFFER
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.renderVertexBuffer);

      // Set geometry (i.e. two triangles forming a rectangle)
      this.gl.bufferData(
         this.gl.ARRAY_BUFFER,
         this.vertices,
         this.gl.STATIC_DRAW
      );
      this.gl.vertexAttribPointer(
         this.renderVertexAttribute,
         2,
         this.gl.FLOAT,
         false,
         0,
         0
      );
   }

   #setupWorld() {
      // Initialize the world with one big sphere for the ground and three
      // smaller spheres for demoing different materials, then randomly place
      // even smaller spheres around those
      let number_of_spheres = 4;

      let material_ground = new Material(
         0,
         glMatrix.vec3.fromValues(0.5, 0.5, 0.5),
         0,
         0
      );
      let material1 = new Material(
         2,
         glMatrix.vec3.fromValues(0, 0, 0),
         0,
         1.5
      );
      let material2 = new Material(
         0,
         glMatrix.vec3.fromValues(0.4, 0.2, 0.1),
         0,
         0
      );
      let material3 = new Material(
         1,
         glMatrix.vec3.fromValues(0.7, 0.6, 0.5),
         0,
         0
      );

      let sphere1 = new Sphere(
         glMatrix.vec3.fromValues(0, -1000, 0),
         1000,
         material_ground
      );
      let sphere2 = new Sphere(glMatrix.vec3.fromValues(0, 1, 0), 1, material1);
      let sphere3 = new Sphere(
         glMatrix.vec3.fromValues(-4, 1, 0),
         1,
         material2
      );
      let sphere4 = new Sphere(glMatrix.vec3.fromValues(4, 1, 0), 1, material3);

      let world = `
      spheres[0] = ${sphere1.toString()};
      spheres[1] = ${sphere2.toString()};
      spheres[2] = ${sphere3.toString()};
      spheres[3] = ${sphere4.toString()};
   `;

      let side = 3;
      number_of_spheres = 4 * side * side + 4;

      var i = 4;
      for (let a = -side; a < side; a++) {
         for (let b = -side; b < side; b++) {
            let sphere_string = "";
            let sphere;
            let mat;
            let choose_mat = Math.random();
            let center = glMatrix.vec3.fromValues(
               a + 0.9 * Math.random(),
               0.2,
               b + 0.9 * Math.random()
            );

            let scene_center = glMatrix.vec3.fromValues(4, 0.2, 0);
            let distance = glMatrix.vec3.create();
            glMatrix.vec3.subtract(distance, center, scene_center);
            // Randomly place spheres of various materials around the scene
            if (glMatrix.vec3.length(distance) > 0.9) {
               if (choose_mat < 0.8) {
                  // diffuse
                  let color = [
                     Math.random() * Math.random(),
                     Math.random() * Math.random(),
                     Math.random() * Math.random(),
                  ];
                  let albedo = glMatrix.vec3.clone(color);
                  mat = new Material(0, albedo, 0, 0);
               } else if (choose_mat < 0.95) {
                  // metal
                  let color = [
                     Math.random() * 0.5,
                     Math.random() * 0.5,
                     Math.random() * 0.5,
                  ];
                  let albedo = glMatrix.vec3.clone(color);
                  let fuzz = Math.random() * 0.5;
                  mat = new Material(1, albedo, fuzz, 0);
               } else {
                  // glass
                  mat = new Material(2, glMatrix.vec3.create(), 0, 1.5);
               }

               sphere = new Sphere(center, 0.2, mat);
               sphere_string = `spheres[${i}] = ${sphere.toString()};
            `;

               world += sphere_string;

               i++;
            }
         }
      }
      return [number_of_spheres, world];
   }

   drawToTexture() {
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures[0]);
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
      // Attach the texture as the first color attachment
      this.gl.framebufferTexture2D(
         this.gl.FRAMEBUFFER,
         this.gl.COLOR_ATTACHMENT0,
         this.gl.TEXTURE_2D,
         this.textures[1],
         0
      );

      // Use textureProgram
      this.gl.useProgram(this.textureProgram);
      this.gl.bindVertexArray(this.textureVao);

      // set uniforms
      this.gl.uniform1f(this.timeLocation, this.timeSinceStart);
      this.gl.uniform1f(
         this.textureWeightLocation,
         this.sampleCount / (this.sampleCount + 1)
      );
      this.gl.uniform3fv(this.eyeLocation, this.cameraPos);
      this.gl.uniform2fv(this.resolutionLocation, this.resolution);

      // Draw texture and unbind framebuffer
      this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
      // "Ping pong" textures
      this.textures.reverse();

      this.sampleCount += 1;
   }

   render() {
      // Render to the canvas
      this.gl.useProgram(this.renderProgram);
      this.gl.bindVertexArray(this.renderVao);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures[0]);
      this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
   }

   tick() {
      this.drawToTexture();
      this.render();

      this.timeSinceStart += 0.0001;
      requestAnimationFrame(() => this.tick());
   }

   rotateCamera() {
      // Rotate camera around x and y axes
      // glMatrix.vec3.rotateY(
      //    this.cameraPos,
      //    this.cameraPos,
      //    this.lookat,
      //    degreesToRadians(angleY)
      // );

      // glMatrix.vec3.rotateX(
      //    this.cameraPos,
      //    this.cameraPos,
      //    this.lookat,
      //    degreesToRadians(angleX)
      // );

      this.cameraPos[0] = zoom * Math.cos(angleX) * Math.sin(angleY);
      this.cameraPos[1] = zoom * Math.cos(angleY);
      this.cameraPos[2] = zoom * Math.sin(angleX) * Math.sin(angleY);

      // Stop the camera from going "beneath" the ground
      // if (this.cameraPos[1] < 0.1) {
      //    this.cameraPos[1] = 0.1;
      // }

      // console.log(
      //    "x: " +
      //       this.cameraPos[0] +
      //       " y: " +
      //       this.cameraPos[1] +
      //       " z: " +
      //       this.cameraPos[2]
      // );

      // Reset samplecount to re-render the texture
      this.sampleCount = 0;
   }

   zoomCamera(amount) {
      // Scale camera position along the vector going from the camera position to the point
      // that is being looked at
      // glMatrix.vec3.scale(this.cameraPos, this.cameraPos, amount);
      zoom *= amount;
      this.cameraPos[0] = zoom * Math.cos(angleX) * Math.sin(angleY);
      this.cameraPos[1] = zoom * Math.cos(angleY);
      this.cameraPos[2] = zoom * Math.sin(angleX) * Math.sin(angleY);
      this.sampleCount = 0;
   }
}

//-------------------------------------------------------------------
function onpointerdown(e) {
   drag = true;
   e.target.style.cursor = "grabbing";
}

function onpointermove(e, renderer) {
   if (drag) {
      angleX += degreesToRadians(-e.movementX);
      angleY += degreesToRadians(-e.movementY);

      angleY = Math.max(angleY, 0.1);
      angleY = Math.min(angleY, Math.PI / 2 - 0.01);

      // console.log(
      //    "angleX: " +
      //       radiansToDegrees(angleX) +
      //       " angleY: " +
      //       radiansToDegrees(angleY) +
      //       " movementX: " +
      //       e.movementX +
      //       " movementY: " +
      //       e.movementY
      // );

      renderer.rotateCamera();
      // console.log(angleX);
   }
}

function onpointerup(e) {
   drag = false;
   e.target.style.cursor = "grab";
}

function onwheel(e, renderer) {
   e.preventDefault();

   var scalingFactor = 0.2;
   renderer.zoomCamera(
      e.wheelDelta < 0 ? 1 - scalingFactor : 1 + scalingFactor
   );
}
//-------------------------------------------------------------------

var drag = false;
var angleX = 0.15;
var angleY = 1.35;
var zoom = 13.5;

function main() {
   var canvas = document.querySelector("#canvas");
   // var gl = WebGLDebugUtils.makeDebugContext(
   //    canvas.getContext("webgl2"),
   //    undefined,
   //    logGLCall
   // );
   var gl = canvas.getContext("webgl2");
   if (!gl) {
      console.log("Your browser does not seem to support webGL");
      return;
   }

   // Set canvas size
   resizeCanvasToDisplaySize(gl.canvas);

   // Set viewport
   gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

   const start = new Date();
   var timeSinceStart = new Date() - start;
   var renderer = new Renderer(
      gl,
      timeSinceStart * 0.001,
      gl.canvas.width,
      gl.canvas.height
   );

   canvas.addEventListener("pointerdown", onpointerdown, false);
   canvas.addEventListener(
      "pointermove",
      (e) => onpointermove(e, renderer),
      false
   );
   canvas.addEventListener("pointerup", onpointerup, false);
   canvas.addEventListener("wheel", (e) => onwheel(e, renderer), {
      passive: false,
   });

   requestAnimationFrame(() => renderer.tick());
}

window.onload = main;
