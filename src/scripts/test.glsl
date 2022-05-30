  precision mediump float;
   varying vec4 v_color;

   // Image
   float aspect_ratio = 16.0 / 9.0;
   float image_width = 600.0;
   float image_height = image_width / aspect_ratio;
   float viewport_height = 2.0;
   float viewport_width = viewport_height * aspect_ratio;
   float focal_length = 1.0;
   int samples_per_pixel = 100;

   // Camera
   vec3 eye = vec3(0.0, 0.0, 0.0);
   vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
   vec3 vertical = vec3(0.0, viewport_height, 0.0);
   vec3 lower_left_corner = eye - horizontal/2.0 - vertical/2.0 - vec3(0.0, 0.0, focal_length);

   // Constants
   float infinity = 10000.0;
   float pi = 3.1415926535897932385;

   float degrees_to_radians(float degrees) {
      return degrees * pi / 180.0;
   }

   // Pseudo-random function taken from https://thebookofshaders.com/10/
   float rand(){
      return fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453);
   }

   float rand(float min, float max) {
      return min + (max-min)*rand();
   }

   struct Ray {
      vec3 origin;
      vec3 direction;
   };

   vec3 at(Ray r, float t) {
      return r.origin + t*r.direction;
   }  

   struct Hit_record {
      vec3 p;
      vec3 normal;
      float t;
      bool front_face;
   };

   void set_face_normal(out Hit_record rec, Ray r, vec3 outward_normal) {
      rec.front_face = dot(r.direction, outward_normal) < 0.0;
      rec.normal = rec.front_face ? outward_normal: -outward_normal;
   }

   struct Sphere {
      vec3 center;
      float radius;
   };

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

   vec3 ray_color(Ray r, Sphere spheres[2]) {
      Hit_record rec;

      if(hit(spheres, r, 0.0, infinity, rec)) {
         return 0.5 * (rec.normal + vec3(1.0, 1.0, 1.0));
      }
      // float t = hit_sphere(vec3(0.0, 0.0, -1.0), 0.5, r);
      // if(t > 0.0) {
      //    vec3 N = normalize(at(r, t) - vec3(0.0, 0.0, -1.0));
      //    return 0.5*vec3(N.x + 1.0, N.y + 1.0, N.z + 1.0);
      // }
      vec3 unit_direction = normalize(r.direction);
      float t = 0.5*(unit_direction.y + 1.0);
      return mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
   }

   // World
   
   void main() {
      // World
      Sphere spheres[2];
      Sphere sphere = Sphere(vec3(0.0, 0.0, -1.0), 0.5);
      spheres[0] = Sphere(vec3(0.0, 0.0, -1.0), 0.5);
      spheres[1] = Sphere(vec3(0.0, -100.5, -1.0), 100.0);

      
      float scale = 1.0 / samples_per_pixel;
      vec3 color = vec3(0.0, 0.0, 0.0);
      for(int s = 0; s < samples_per_pixel; ++s) {
         float u = (gl_FragCoord.x + rand()) / (image_width - 1.0);
         float v = (gl_FragCoord.y + rand()) / (image_height - 1.0);
         Ray r = Ray(eye, lower_left_corner + u*horizontal + v*vertical - eye);
         
         color += clamp(scale*ray_color(r, spheres), 0.0, 0.999);
      }
      gl_FragColor = vec4(color, 1.0);
   }
)