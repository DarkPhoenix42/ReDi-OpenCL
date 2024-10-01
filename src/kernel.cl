struct Cell {
  float a, b, lap_a, lap_b;
  int last_c;
};

typedef struct Cell KCell;

__kernel void update(__global KCell *kcells, float D_A, float D_B, float FEED,
                     float KILL) {

  const int pos = get_global_id(0);
  kcells[pos].a +=
      (D_A * kcells[pos].lap_a - kcells[pos].a * kcells[pos].b * kcells[pos].b +
       FEED * (1 - kcells[pos].a));

  kcells[pos].b +=
      (D_B * kcells[pos].lap_b + kcells[pos].a * kcells[pos].b * kcells[pos].b -
       (FEED + KILL) * kcells[pos].b);
}

__kernel void calculate_laplacian(__global KCell *kcells, int WIDTH, int HEIGHT,
                                  __global float *k_weights) {

  const int pos = get_global_id(0);
  const int row = pos / WIDTH, column = pos % WIDTH;

  kcells[pos].lap_a = -kcells[pos].a;
  kcells[pos].lap_b = -kcells[pos].b;

  int i = 0;
  for (int r = row - 1; r < row + 2; r++) {
    for (int c = column - 1; c < column + 2; c++) {
      
      if ((row == r && column == c)) {
        continue;
      }

      i++;

      if (r >= 0 && c >= 0 && r < HEIGHT && c < WIDTH) {
        int neighbour_pos = r * WIDTH + c;
        kcells[pos].lap_a += (kcells[neighbour_pos].a * k_weights[i]);
        kcells[pos].lap_b += (kcells[neighbour_pos].b * k_weights[i]);
      }
    }
  }
}
