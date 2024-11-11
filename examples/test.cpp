#include "../include/BPM/BPM.hpp"

// NOTE: There is no logic to be found in this example
//       This example is only to ensure that the calls work

int main()
{
    bpm::Vec3 a(0, 1, 0);
    bpm::Vec3 b(1, 0, 1);

    bpm::Vec3 c = (a + b) / 2;
    c = normalize(c);

    bpm::Mat4 mat = bpm::Mat4::look_at({ -2, 2, -2 }, { 0, 0, 0 }, { 0, 1, 0 });
    bpm::Vec4 d = bpm::transform(bpm::Vec4(c), mat);
    d = bpm::lerp(d, bpm::Vec4(1), 0.5f);

    bpm::Vec3 e(d[0], d[1], d[2]);
    bpm::Quat q1(1.0f, 1.0f, 1.0f);
    bpm::Quat q2(0.5f, 0.5f, 0.5f);
    bpm::Quat q = bpm::nlerp(q1, q2, 0.5);
    bpm::rotate(e, q);

    std::cout << bpm::IVec4(abs(d)) << std::endl;
}
