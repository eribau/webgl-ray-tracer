gl.shaderSource([object WebGLShader], #version 300 es
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

   float degrees_to_radians(float degrees) {
      return degrees * pi / 180.0;
   }

   // Functions for generating pseudorandom numbers
   // Taken from https://www.shadertoy.com/view/llVcDz
   float g_seed = 0.25;

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

   struct Ray {
      vec3 origin;
      vec3 direction;
   };

   vec3 at(Ray r, float t) {
      return r.origin + t*r.direction;
   }  

   struct Camera {
      vec3 origin;
      vec3 horizontal;
      vec3 vertical;
      vec3 lower_left_corner;
   };

   Ray get_ray(Camera c, float u, float v) {
      return Ray(c.origin, normalize(c.lower_left_corner + u*c.horizontal + v*c.vertical - c.origin));
   }

   struct Material {
      int material; // 0 = lambertian, 1 = metal
      vec3 albedo;
   };

   struct Hit_record {
      vec3 p;
      vec3 normal;
      float t;
      bool front_face;
      Material material;
   };

   void set_face_normal(out Hit_record rec, Ray r, vec3 outward_normal) {
      rec.front_face = dot(r.direction, outward_normal) < 0.0;
      rec.normal = rec.front_face ? outward_normal: -outward_normal;
   }

   bool lambertian_scatter(in Hit_record rec, inout vec3 attenuation, inout Ray scattered) {
      vec3 scatter_direction = random_in_hemisphere(rec.normal);
      scattered = Ray(rec.p, scatter_direction);
      attenuation = rec.albedo;
      return true;
   }

   struct Sphere {
      vec3 center;
      float radius;
      Material material;
   };

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

   bool hit(Sphere spheres[2], Ray r, float t_min, float t_max, out Hit_record rec) {
      Hit_record temp_rec;
      bool hit_anything = false;
      float closest_so_far = t_max;

      for(int i = 0; i < 2; i++) {
         // TODO Rewrite hit_sphere to use Sphere instead of center and radius 
         if(hit_sphere(spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
         }
      }

      return hit_anything;
   }

   vec3 ray_color(Ray r, Sphere spheres[2]) {
      Hit_record rec;
      vec3 color = vec3(1.0);

      for(int i = 0; i < max_depth; i++) {
         if(hit(spheres, r, 0.001, infinity, rec)) {
            // vec3 target = rec.normal + normalize(random_in_unit_sphere(g_seed));
            // vec3 target = random_in_hemisphere(rec.normal);
            // color *= 0.5;

            // r.origin = rec.p;
            // r.direction = target;

            switch (rec.material.material) {
               case 0:
                  Ray scattered;
                  vec3 attenuation;
                  if(lambertian_scatter(rec, scattered, attenuation)) {
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

   void main() {
      vec2 resolution = vec2(600, 338);
      float aspect = resolution.x / resolution.y;

      // Materials
      Material material_ground = Material(0, vec3(0.8, 0.8, 0.0));
      Material material_center = Material(0, vec3(0.7, 0.3, 0.3));

      // World
      Sphere spheres[2];
      spheres[0] = Sphere(vec3(0.0, 0.0, -1.0), 0.5, material_center);
      spheres[1] = Sphere(vec3(0.0, -100.5, -1.0), 100.0, material_ground);

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
)