#pragma once
#include <iostream>
#include <map>

using namespace std;

enum Shape
{
    CIRCLE = 0,
    SQUARE = 1
};

inline uint8_t Clamp0255(const int c)
{
    return (c < 0) ? 0 : ((c > 255) ? 255 : c);
}

inline float d_sq(float x1, float y1, float x2, float y2)
{
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

inline bool is_point_in_circle(float x, float y, float c_x, float c_y, float r)
{
    return (d_sq(x, y, c_x, c_y) < r * r);
}

const char scroll = 27;

void print_stats(map<string, int *> *stats)
{
    cout << scroll << "[2J\n";
    cout << scroll << "[1;1f";

    for (map<string, int *>::iterator stat = stats->begin(); stat != stats->end(); ++stat)
        cout << stat->first << *(stat->second) << endl;

    cout.flush();
}
