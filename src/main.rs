use image::{ImageBuffer, Rgb};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

// ===================== Math / Vec =====================
#[derive(Clone, Copy, Debug, Default)]
struct Vec3 { x: f32, y: f32, z: f32 }
impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    fn add(self, o: Vec3) -> Self { Self::new(self.x+o.x, self.y+o.y, self.z+o.z) }
    fn sub(self, o: Vec3) -> Self { Self::new(self.x-o.x, self.y-o.y, self.z-o.z) }
    fn mul(self, s: f32) -> Self { Self::new(self.x*s, self.y*s, self.z*s) }
    fn dot(self, o: Vec3) -> f32 { self.x*o.x + self.y*o.y + self.z*o.z }
    fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(self.y*o.z - self.z*o.y, self.z*o.x - self.x*o.z, self.x*o.y - self.y*o.x)
    }
    fn norm(self) -> f32 { self.dot(self).sqrt() }
    fn normalize(self) -> Self { let n = self.norm(); if n>0.0 { self.mul(1.0/n) } else { self } }
}
fn rodrigues(v: Vec3, k: Vec3, theta: f32) -> Vec3 {
    let (ct, st) = (theta.cos(), theta.sin());
    v.mul(ct).add(k.cross(v).mul(st)).add(k.mul(k.dot(v)*(1.0-ct)))
}
fn rotate_towards(v: Vec3, t: Vec3, ang: f32) -> Vec3 {
    let v = v.normalize();
    let t = t.normalize();
    let cos_g = v.dot(t).clamp(-1.0, 1.0);
    let g = cos_g.acos();
    if g < 1e-6 { return v; }
    let axis = v.cross(t);
    let n = axis.norm();
    if n < 1e-6 {
        let k_alt = if v.x.abs() < 0.9 { Vec3::new(1.0,0.0,0.0) } else { Vec3::new(0.0,1.0,0.0) };
        let k2 = v.cross(k_alt).normalize();
        return rodrigues(v, k2, ang.min(g*0.999)).normalize();
    }
    rodrigues(v, axis.mul(1.0/n), ang.min(g*0.999)).normalize()
}

// ===================== Noise / FBM =====================
fn hash01_3(p: Vec3, seed: u32) -> f32 {
    let k1 = 127.1 + (seed as f32)*0.0001;
    let k2 = 311.7 + (seed as f32)*0.0003;
    let k3 =  74.7 + (seed as f32)*0.0007;
    let h = (p.x*k1 + p.y*k2 + p.z*k3).sin() * 43758.5453;
    h.fract().abs()
}
fn s_curve(t: f32) -> f32 { t*t*(3.0 - 2.0*t) } // smoothstep
fn val_noise2(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let p = |ix: f32, iy: f32| hash01_3(Vec3::new(ix, iy, ix+iy), seed);
    let v00 = p(xi, yi);
    let v10 = p(xi+1.0, yi);
    let v01 = p(xi, yi+1.0);
    let v11 = p(xi+1.0, yi+1.0);
    let u = s_curve(xf);
    let v = s_curve(yf);
    let a = v00*(1.0-u) + v10*u;
    let b = v01*(1.0-u) + v11*u;
    a*(1.0-v) + b*v
}
fn fbm2(mut x: f32, mut y: f32, seed: u32, oct: i32, lac: f32, gain: f32) -> f32 {
    let mut amp = 0.5;
    let mut sum = 0.0;
    for _ in 0..oct {
        sum += amp * val_noise2(x, y, seed);
        x *= lac; y *= lac;
        amp *= gain;
    }
    sum
}

// ===================== Color utils =====================
fn mix3(a: [f32;3], b: [f32;3], t: f32) -> [f32;3] {
    [a[0]*(1.0-t)+b[0]*t, a[1]*(1.0-t)+b[1]*t, a[2]*(1.0-t)+b[2]*t]
}
fn hsv(h: f32, s: f32, v: f32) -> [f32;3] {
    let h = (h % 1.0 + 1.0) % 1.0;
    let i = (h*6.0).floor();
    let f = h*6.0 - i;
    let p = v*(1.0-s);
    let q = v*(1.0-s*f);
    let t = v*(1.0-s*(1.0-f));
    match (i as i32) % 6 {
        0 => [v, t, p], 1 => [q, v, p], 2 => [p, v, t],
        3 => [p, q, v], 4 => [t, p, v], _ => [v, p, q],
    }
}

// ACES-ish tonemap (simple approximation) then gamma 2.2
fn tonemap_aces(x: f32) -> f32 {
    // ACES fitted curve
    let a=2.51; let b=0.03; let c=2.43; let d=0.59; let e=0.14;
    let y = ((x*(a*x+b))/(x*(c*x+d)+e)).clamp(0.0, 1.0);
    y.powf(1.0/2.2)
}
fn to_u8(x: f32) -> u8 { (tonemap_aces(x) * 255.0 + 0.5) as u8 }

// ===================== Scene Shaders =====================
fn sample_nebula(dir: Vec3, seed: u32) -> [f32;3] {
    // Nebula background with FBM + hue drift; keeps stars subtle
    let t = (dir.z*0.5 + 0.5).clamp(0.0, 1.0);
    let grad = mix3([0.01,0.015,0.035],[0.06,0.07,0.12], t);

    let u = fbm2(dir.x*3.1, dir.y*3.7 + dir.z*1.9, seed, 5, 2.0, 0.55);
    let v = fbm2(dir.x*6.3+11.0, dir.y*5.1+3.0, seed.wrapping_add(77), 4, 2.2, 0.5);
    let cloud = ((u*0.8 + v*0.6) - 0.4).max(0.0);

    let hue = (u*0.35 + v*0.25 + 0.55).fract();
    let neb = hsv(hue, 0.6, 0.8);
    mix3(grad, [neb[0]*cloud, neb[1]*cloud, neb[2]*cloud], 0.85)
}

fn sample_swirl_ring(u: f32, v: f32, b: f32, b_crit: f32, rs: f32, seed: u32) -> [f32;3] {
    let band = 0.65 * rs;                  // ring thickness
    let d = (b - b_crit) / band;
    let radial = (1.0 - d.abs()).clamp(0.0, 1.0);
    if radial <= 0.0 { return [0.0,0.0,0.0]; }

    let angle = v.atan2(u);
    let base_h = (angle / (2.0*std::f32::consts::PI)) + 0.5;

    let n1 = val_noise2(u*8.0, v*8.0, seed);
    let n2 = val_noise2(u*16.0+3.0, v*16.0-1.0, seed.wrapping_add(1234));
    let swirl = 0.28 * ( (6.0*angle).sin() ) + 0.22*(n1 - 0.5) + 0.12*(n2 - 0.5);
    let hue = (base_h + swirl).fract();

    // brighter near center of band
    let s = 0.85 + 0.15*radial;
    let vv = 0.9 + 0.9*radial;
    let c = hsv(hue, s, vv);
    [c[0]*radial, c[1]*radial, c[2]*radial]
}

fn sample_disk(pos: Vec3, rs: f32, r_in: f32, r_out: f32, seed: u32) -> ([f32;3], bool) {
    let r = (pos.x*pos.x + pos.y*pos.y).sqrt();
    if r < r_in || r > r_out { return ([0.0,0.0,0.0], false); }

    let t = ((r - r_in) / (r_out - r_in)).clamp(0.0,1.0);
    let base = mix3([1.2,0.9,0.45],[0.5,0.7,1.1], t);
    let phi = pos.y.atan2(pos.x);
    let n = val_noise2(phi.cos()*8.0 + 3.0, phi.sin()*8.0 - 1.3, seed);
    let bands = ( (18.0*phi + 9.0*n).sin()*0.5 + 0.5 ) * 0.35;
    let glow = 0.9 + 0.7*(1.0 - t).powf(0.7);
    let c = [
        (base[0]*glow + bands).clamp(0.0, 2.0),
        (base[1]*glow + 0.5*bands).clamp(0.0, 2.0),
        (base[2]*glow).clamp(0.0, 2.0),
    ];
    (c, true)
}

fn doppler_boost(beta: f32, mu: f32) -> f32 {
    let beta = beta.clamp(0.0, 0.95);
    let gamma = 1.0 / (1.0 - beta*beta).sqrt();
    let delta = 1.0 / (gamma * (1.0 - beta * mu).max(0.05));
    delta*delta*delta
}

// ===================== Post FX =====================
// simple separable gaussian kernel (weights sum ~1)
fn gaussian_weights(radius: usize, sigma: f32) -> Vec<f32> {
    let mut w = vec![0.0; radius*2+1];
    let s2 = 2.0*sigma*sigma;
    let mut sum = 0.0;
    for i in 0..w.len() {
        let x = i as isize - radius as isize;
        let val = (-((x as f32)*(x as f32))/s2).exp();
        w[i] = val; sum += val;
    }
    for v in w.iter_mut() { *v /= sum.max(1e-6); }
    w
}

fn bloom(bright: &[[f32;3]], w: usize, h: usize, radius: usize, sigma: f32) -> Vec<[f32;3]> {
    let weights = gaussian_weights(radius, sigma);
    let mut tmp = vec![[0.0;3]; w*h];
    let mut out = vec![[0.0;3]; w*h];

    // horizontal
    tmp.par_iter_mut().enumerate().for_each(|(i, t)| {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        let mut acc = [0.0f32;3];
        for k in 0..weights.len() {
            let dx = k as isize - radius as isize;
            let xx = (x + dx).clamp(0, w as isize - 1) as usize;
            let s = bright[y as usize * w + xx];
            let wgt = weights[k];
            acc[0]+=s[0]*wgt; acc[1]+=s[1]*wgt; acc[2]+=s[2]*wgt;
        }
        *t = acc;
    });

    // vertical
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        let mut acc = [0.0f32;3];
        for k in 0..weights.len() {
            let dy = k as isize - radius as isize;
            let yy = (y + dy).clamp(0, h as isize - 1) as usize;
            let s = tmp[yy * w + x as usize];
            let wgt = weights[k];
            acc[0]+=s[0]*wgt; acc[1]+=s[1]*wgt; acc[2]+=s[2]*wgt;
        }
        *o = acc;
    });
    out
}

// ===================== Main =====================
fn main() {
    // ---------- Tunables ----------
    let width = 1280usize;
    let height = 720usize;

    // supersample scale (2 = 2x width/height; crisp AA)
    let ss = 2usize;
    let rw = width*ss;
    let rh = height*ss;

    // camera & black hole params
    let fov_deg = 72.0f32;
    let rs = 1.0f32;
    let cam_dist = 9.0f32 * rs;
    let tilt_deg = 9.0f32; // artistic camera roll
    let enable_disk = true;
    let r_in = 3.0*rs;
    let r_out = 12.0*rs;

    // post FX
    let bloom_threshold = 1.2f32;
    let bloom_radius_px = 10usize;
    let bloom_sigma = 6.0f32;
    let vignette_strength = 0.32f32;

    // random seed per run (set fixed value for reproducibility)
    let mut rng = rand::thread_rng();
    let seed_u64: u64 = rng.gen();
    let seed32: u32 = (seed_u64 ^ (seed_u64>>32)) as u32;

    // Camera basis with roll
    let aspect = rw as f32 / rh as f32;
    let fov = fov_deg.to_radians();
    let half_w = (fov*0.5).tan();
    let half_h = half_w / aspect;

    let forward = Vec3::new(0.0, 0.0, -1.0);
    let up0 = Vec3::new(0.0, 1.0, 0.0);
    let right0 = up0.cross(forward).normalize();
    let roll = tilt_deg.to_radians();
    // rotate the (right,up) frame around forward to get camera roll
    let right = rodrigues(right0, forward, roll).normalize();
    let up = rodrigues(up0, forward, roll).normalize();

    let cam_pos = Vec3::new(0.0, 0.0, cam_dist);
    let b_crit = (3.0f32.sqrt() * 1.5) * rs;

    // render buffer (HDR linear)
    let mut hdr = vec![[0.0f32;3]; rw*rh];

    hdr.par_iter_mut().enumerate().for_each(|(idx, px)| {
        let x = (idx % rw) as i32;
        let y = (idx / rw) as i32;

        // NDC
        let u = ((x as f32 + 0.5)/rw as f32)*2.0 - 1.0;
        let v = ((y as f32 + 0.5)/rh as f32)*2.0 - 1.0;

        // primary ray (thin lens)
        let sx = u*half_w;
        let sy = -v*half_h;
        let dir0 = (forward.add(right.mul(sx)).add(up.mul(sy))).normalize();

        // impact parameter estimate
        let radial = forward;
        let cos_psi = dir0.dot(radial).clamp(-1.0, 1.0);
        let sin_psi = (1.0 - cos_psi*cos_psi).sqrt();
        let b = cam_dist * sin_psi;

        // black hole shadow
        if b < b_crit {
            *px = [0.0,0.0,0.0];
            return;
        }

        // chromatic slight offsets for prismatic Einstein ring
        let bend = |alpha_scale: f32| -> Vec3 {
            let mut alpha = 2.0 * rs / (b + 1e-6) * alpha_scale;
            alpha = alpha.min(1.2);
            rotate_towards(dir0, radial, alpha)
        };
        let dir_r = bend(1.00);
        let dir_g = bend(1.03);
        let dir_b = bend(1.06);

        // Nebula background
        let mut col = [
            sample_nebula(dir_r, seed32)[0],
            sample_nebula(dir_g, seed32)[1],
            sample_nebula(dir_b, seed32)[2],
        ];

        // Swirling rainbow ring hugging photon ring
        let swirl = sample_swirl_ring(u, v, b, b_crit, rs, seed32);
        col[0] += swirl[0];
        col[1] += swirl[1];
        col[2] += swirl[2];

        // Optional thin accretion disk (lensed by bent ray intersection with z=0)
        if enable_disk {
            let add_disk = |dir: Vec3, ch: usize, acc: &mut [f32;3]| {
                if dir.z >= 0.0 { return; }
                let t = -cam_pos.z / dir.z;
                if t <= 0.0 { return; }
                let hit = cam_pos.add(dir.mul(t));
                let (mut dcol, ok) = sample_disk(hit, rs, r_in, r_out, seed32);
                if !ok { return; }

                let r = (hit.x*hit.x + hit.y*hit.y).sqrt().max(r_in);
                let beta = (rs / (2.0*r)).sqrt().min(0.6);
                let phi = hit.y.atan2(hit.x);
                let vhat = Vec3::new(-phi.sin(), phi.cos(), 0.0);
                let mu = vhat.dot(dir.mul(-1.0).normalize()).clamp(-1.0, 1.0);

                let boost = doppler_boost(beta, mu);
                let ggrav = (1.0 - rs / r).max(0.1).sqrt();
                dcol[0]*=boost*ggrav; dcol[1]*=boost*ggrav; dcol[2]*=boost*ggrav;

                // self-occlusion
                let occl = ((r - r_in)/(r_out - r_in)).clamp(0.0,1.0);
                dcol = [dcol[0]*occl, dcol[1]*occl, dcol[2]*occl];

                acc[ch] = (acc[ch] + dcol[ch]).max(acc[ch]);
            };
            add_disk(dir_r, 0, &mut col);
            add_disk(dir_g, 1, &mut col);
            add_disk(dir_b, 2, &mut col);
        }

        // ring contrast shading
        let ring_soft = ((b - b_crit) / (0.6*rs)).clamp(0.0, 1.0);
        let shade = 0.60 + 0.40*ring_soft;
        col[0]*=shade; col[1]*=shade; col[2]*=shade;

        // vignette
        let r2 = (u*u + v*v).min(1.0);
        let vignette = 1.0 - vignette_strength * r2.powf(1.15);
        col[0]*=vignette; col[1]*=vignette; col[2]*=vignette;

        // store HDR
        *px = col;
    });

    // --------- Bloom (bright pass + blur) ---------
    let mut bright = vec![[0.0f32;3]; rw*rh];
    bright.par_iter_mut().enumerate().for_each(|(i, b)| {
        let c = hdr[i];
        let l = (0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]).max(0.0);
        let k = (l - bloom_threshold).max(0.0);
        *b = [c[0]*k, c[1]*k, c[2]*k];
    });
    let bloom_blur = bloom(&bright, rw, rh, bloom_radius_px, bloom_sigma);

    // add bloom back
    hdr.par_iter_mut().enumerate().for_each(|(i, c)| {
        let b = bloom_blur[i];
        c[0] += b[0]; c[1] += b[1]; c[2] += b[2];
    });

    // --------- Downsample (SSAA 2x) + grain dithering ---------
    let mut ldr = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    let inv = 1.0 / ((ss*ss) as f32);
    let mut rng = rand::rngs::StdRng::from_seed([42u8;32]); // stable grain pattern
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32;3];
            for oy in 0..ss {
                for ox in 0..ss {
                    let sx = x*ss + ox;
                    let sy = y*ss + oy;
                    let c = hdr[sy*rw + sx];
                    acc[0]+=c[0]; acc[1]+=c[1]; acc[2]+=c[2];
                }
            }
            let mut c = [acc[0]*inv, acc[1]*inv, acc[2]*inv];

            // subtle film grain (pre-tonemap) to break up banding
            let n: f32 = rng.gen::<f32>() - 0.5;
            let g = n * 0.02; // grain strength
            c[0] = (c[0] + g).max(0.0);
            c[1] = (c[1] + g).max(0.0);
            c[2] = (c[2] + g).max(0.0);

            ldr.put_pixel(x as u32, y as u32, Rgb([to_u8(c[0]), to_u8(c[1]), to_u8(c[2])]));
        }
    }

    ldr.save("blackhole.png").expect("failed to save blackhole.png");
    println!("Rendered blackhole.png  ({}x{}, SSAA {}x, seed {})", width, height, ss, seed32);
}
