#include <iostream>
#include <fstream>
#include "../include/json.hpp"
#include "../include/argparse.hpp"
#include <SDL2/SDL.h>

#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "Utils.cpp"

using namespace std;

// Parameters
int WIDTH, HEIGHT, INIT_SHAPE, INIT_SIZE, FRAMES_TO_RENDER;
float FEED, KILL;
const float weights[8] = {0.05, 0.2, 0.05, 0.2, 0.2, 0.05, 0.2, 0.05}, D_A = 1.0, D_B = 0.5;

// Stats
int frame = 0, fps = 0, fps_counter = 0;
map<string, int *> stats{{"1. FPS: ", &fps}, {"2. Frame: ", &frame}};
Uint32 fps_timer, draw_timer;

// SDL Stuff
SDL_Renderer *renderer;
SDL_Window *win;
SDL_Texture *texture;
SDL_Event event;

SDL_PixelFormat *format;
Uint32 *pixels;
SDL_Color color;
int pitch;

// OpenCL Stuf
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

cl_kernel laplace_kernel;
cl_kernel update_kernel;

cl_mem k_cells;
cl_mem k_weights;

size_t global_work_size[1];

struct Cell
{
    float a = 1, b = 0, lap_a = 0, lap_b = 0;

    void seed_b()
    {
        b = 1.0;
    }

    void draw(int pos)
    {
        const int c = Clamp0255((a - b) * 255);
        pixels[pos] = SDL_MapRGB(format, Clamp0255(color.r - c), Clamp0255(color.g - c), Clamp0255(color.b - c));
    }
};

Cell *cells;

void update()
{

    clEnqueueWriteBuffer(queue, k_cells, CL_TRUE, 0, WIDTH * HEIGHT * sizeof(Cell), cells, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, laplace_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, update_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, k_cells, CL_TRUE, 0, WIDTH * HEIGHT * sizeof(Cell), cells, 0, NULL, NULL);
    clFinish(queue);

    frame++;
    fps_counter++;

    if (frame == FRAMES_TO_RENDER)
        exit(0);

    // Calculating FPS
    if (SDL_GetTicks() - fps_timer > 1000.f)
    {
        fps = fps_counter;
        fps_counter = 0;
        fps_timer = SDL_GetTicks();
    }
    print_stats(&stats);
}

void draw_screen()
{
    SDL_LockTexture(texture, NULL, (void **)&pixels, &pitch);

    for (int pos = 0; pos < WIDTH * HEIGHT; pos++)
        cells[pos].draw(pos);

    SDL_UnlockTexture(texture);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

void handle_events()
{
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
            exit(0);
    }
}

void init_cells()
{

    cells = new Cell[WIDTH * HEIGHT];

    int c_column = floor(WIDTH / 2);
    int c_row = floor(HEIGHT / 2);

    // Seeding the init shape with b
    for (int row = c_row - INIT_SIZE; row < c_row + INIT_SIZE; row++)
    {
        for (int column = c_column - INIT_SIZE; column < c_column + INIT_SIZE; column++)
        {
            int pos = row * WIDTH + column;
            if (INIT_SHAPE == CIRCLE)
            {
                if (is_point_in_circle(column, row, c_column, c_row, INIT_SIZE))
                    cells[pos].seed_b();
            }
            else if (INIT_SHAPE == SQUARE)
                cells[pos].seed_b();
        }
    }
}

void init_OpenCL()
{
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_int status = 0;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    ifstream file("../src/kernel.cl");
    string src(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    const char *src_c = src.c_str();

    program = clCreateProgramWithSource(context, 1, (const char **)&src_c, NULL, &status);
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (status != CL_SUCCESS)
    {
        cout << "Couldn't build program!" << endl;
        size_t len;

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        cout << len << endl;
        char buf[len];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &len);
        cout << buf << endl;
        exit(1);
    }

    laplace_kernel = clCreateKernel(program, "calculate_laplacian", &status);
    if (status != CL_SUCCESS)
    {
        cout << "Couldn't create laplace_kernel!" << endl;
        exit(1);
    }

    update_kernel = clCreateKernel(program, "update", &status);
    if (status != CL_SUCCESS)
    {
        cout << "Couldn't create update_kernel!" << endl;
        exit(1);
    }

    k_cells = clCreateBuffer(context, CL_MEM_READ_WRITE, WIDTH * HEIGHT * sizeof(Cell), NULL, NULL);
    k_weights = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(queue, k_weights, CL_TRUE, 0, 8 * sizeof(float), weights, 0, NULL, NULL);
    clFinish(queue);

    clSetKernelArg(laplace_kernel, 0, sizeof(cl_mem), &k_cells);
    clSetKernelArg(laplace_kernel, 1, sizeof(int), &WIDTH);
    clSetKernelArg(laplace_kernel, 2, sizeof(int), &HEIGHT);
    clSetKernelArg(laplace_kernel, 3, sizeof(cl_mem), &k_weights);

    clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &k_cells);
    clSetKernelArg(update_kernel, 1, sizeof(float), &D_A);
    clSetKernelArg(update_kernel, 2, sizeof(float), &D_B);
    clSetKernelArg(update_kernel, 3, sizeof(float), &FEED);
    clSetKernelArg(update_kernel, 4, sizeof(float), &KILL);

    global_work_size[0] = WIDTH * HEIGHT;
}

void init_SDL()
{
    SDL_Init(SDL_INIT_EVERYTHING);
    win = SDL_CreateWindow("Reaction Diffusion", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_BGR888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    format = SDL_AllocFormat(SDL_PIXELFORMAT_BGR888);
}

void load_args(int argc, char *argv[])
{
    // Configuration
    nlohmann::json config;
    fstream config_file;

    try
    {
        config_file.open("../config.json", ios::in);
    }
    catch (const std::exception &e)
    {
        cout << "Couldn't open config.json file. Exiting..." << endl;
        exit(1);
    }

    config_file >> config;
    config_file.close();

    WIDTH = config["width"];
    HEIGHT = config["height"];
    FRAMES_TO_RENDER = config["frames_to_render"];

    nlohmann::json system = config["systems"][(string)config["defaultSystem"]];
    FEED = system["feed"];
    KILL = system["kill"];
    INIT_SHAPE = system["init_shape"];
    INIT_SIZE = system["init_size"];

    color.r = config["color"].at(0);
    color.g = config["color"].at(1);
    color.b = config["color"].at(2);

    argparse::ArgumentParser parser("Reaction Diffusion Simulator");
    parser.add_description("A program to simulate Reaction Diffusion using the Gray-Scott model. Give initial parameters as arguments or edit the config.json file.");

    parser.add_argument("--width", "-w")
        .default_value(WIDTH)
        .scan<'i', int>()
        .help("Set the width of the simulation.");

    parser.add_argument("--height", "-ht")
        .default_value(HEIGHT)
        .scan<'i', int>()
        .help("Set the height of the simulation.");

    parser.add_argument("--frames_to_render", "-f")
        .default_value(FRAMES_TO_RENDER)
        .scan<'i', int>()
        .help("Set the number of frames to run the simulation.");

    parser.add_argument("--system", "-s")
        .help("Set the initial parameters according to a given system present in config.json.\n\
                Note that any parameters passed as arguments will override the system parameters.")

        .default_value((string)config["defaultSystem"]);

    parser.add_argument("--feed", "-f")
        .default_value(FEED)
        .scan<'g', float>()
        .help("Set the feed rate of the simulation.");

    parser.add_argument("--kill", "-k")
        .default_value(KILL)
        .scan<'g', float>()
        .help("Set the kill rate of the simulation.");

    parser.add_argument("--init_shape", "-ishape")
        .default_value(INIT_SHAPE)
        .scan<'i', int>()
        .help("Set the shape of the area initialised with chemical b.");

    parser.add_argument("--init_size", "-isize")
        .default_value(INIT_SIZE)
        .scan<'i', int>()
        .help("Set the size of the area initialised with chemical b.");

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const runtime_error &err)
    {
        cout << err.what() << endl;
        cout << parser;
        exit(1);
    }

    WIDTH = parser.get<int>("--width");
    HEIGHT = parser.get<int>("--height");
    FRAMES_TO_RENDER = parser.get<int>("--frames_to_render");

    if (!config["systems"].contains(parser.get<string>("--system")))
    {
        cout << "System is not included in config.json. Systems Present: " << endl;
        for (auto system : config["systems"].items())
        {
            cout << system.key() << endl;
        }
        exit(0);
    }

    system = config["systems"][parser.get<string>("--system")];
    FEED = system["feed"];
    KILL = system["kill"];
    INIT_SHAPE = system["init_shape"];
    INIT_SIZE = system["init_size"];

    if (parser.is_used("--feed"))
        FEED = parser.get<float>("--feed");
    if (parser.is_used("--kill"))
        KILL = parser.get<float>("--kill");
    if (parser.is_used("--init_shape"))
        INIT_SHAPE = parser.get<int>("--init_shape");
    if (parser.is_used("--init_size"))
        INIT_SIZE = parser.get<int>("--init_size");
    if (INIT_SHAPE != CIRCLE && INIT_SHAPE != SQUARE)
    {
        cout << "Init Shape must be either 0 or 1 i.e Circle or Square." << endl;
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    cout << "Loading Arguments..." << endl;
    load_args(argc, argv);

    cout << "Initializing Cells..." << endl;
    init_cells();

    cout << "Initializing OpenCL..." << endl;
    init_OpenCL();

    cout << "Initializing SDL..." << endl;
    init_SDL();

    fps_timer = SDL_GetTicks();
    draw_timer = SDL_GetTicks();
    while (1)
    {
        handle_events();

        if (SDL_GetTicks() - draw_timer > (float)(1000 / 60))
        {
            draw_screen();
            draw_timer = SDL_GetTicks();
        }

        update();
    }
}
