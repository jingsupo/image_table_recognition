#include <algorithm>
#include <iostream>
#include <vector>
#include <io.h>//_access
#include <opencv2/opencv.hpp>

#if defined(_MSC_VER)
#include <direct.h>
#define GetCurrentDir _getcwd
#elif defined(__unix__)
#include <unistd.h>
#define GetCurrentDir getcwd
#else
#endif
#include <fstream>

///处理流程：
///1、图像灰度化处理
///2、图像二值化处理
///3、图像腐蚀膨胀处理
///4、获取表格交点坐标
///5、根据交点集获取单元格轮廓并进行过滤
class image_table_recognition
{
public:
    std::vector<std::vector<int>> run(std::string img_path, std::string out_path);

private:
    int params_margin_x = 10;   //x轴点达到一定的距离，才算两个有效点，确定线的个数
    int params_margin_y = 10;   //y轴点达到一定的距离，才算两个有效点，确定线的个数
    int params_dot_margin = 10; //和平均线的偏移量（对缝隙起作用，可去除点，也可变为独立一个点）
    int params_line_x = 10;     //x上点个数的差值调节（线不均匀，有的粗有的细，甚至有的不连续）
    int params_line_y = 10;     //y上点个数的差值调节（线不均匀，有的粗有的细，甚至有的不连续）

private:
    std::pair<std::vector<int>, std::vector<int>> _where(cv::Mat mat)
    {
        std::vector<int> x;
        std::vector<int> y;
        cv::Size size = mat.size();
        for (int i = 0; i < size.width; i++)
        {
            for (int j = 0; j < size.height; j++)
            {
                if (mat.at<int>(i, j) > 0)
                {
                    x.push_back(i);
                    y.push_back(j);
                }
            }
        }
        return std::make_pair(x, y);
    }

    bool _in_vec(std::vector<int> vec, int val)
    {
        return std::find(vec.begin(), vec.end(), val) != vec.end();
    }

    bool _multi_in_vec(std::vector<std::vector<int>> vec, int val, ...)
    {
        //为了实现可变参数列表，首先需要声明一个va_list类型的指针
        //该指针用来依次指向各个参数
        va_list arg_ptr;
        //va_start是一个宏，用来初始化arg_ptr，使其指向列表的第一个参数
        //这个宏的第二个参数是函数参数列表省略号前的固定参数的名称，用来确定第一个参数的位置
        va_start(arg_ptr, val);
        //va_arg是一个宏，返回arg_ptr指向的参数位置，并使arg_ptr递增来指向下一个参数值
        //va_arg宏的第二个参数是第一个参数的类型
        int y = val;
        int x = va_arg(arg_ptr, int);
        //将va_list类型的指针复位，清空可变参数列表
        va_end(arg_ptr);
        auto yx_in = [=](std::vector<int> v) { return (v[0] == y) and (v[1] == x); };
        return std::find_if(vec.begin(), vec.end(), yx_in) != vec.end();
    }

    int _index(std::vector<int> vec, int val)
    {
        auto it = std::find(vec.begin(), vec.end(), val);
        if (it != vec.end())
        {
            return std::distance(vec.begin(), it);
        }
        return -1;
    }

    int _recognize_line(std::vector<int> line_xs, std::vector<int> line1, std::vector<int> line2, int num, int num1, int num2)
    {
        std::vector<int> line_list;
        for (auto& k : { -3, -2, -1, 0, 1, 2, 3 })
        {
            for (int i = 0; i < line_xs.size(); i++)
            {
                if (line1[i] == num + k)
                {
                    if ((num1 <= line2[i]) and (line2[i] <= num2) and (not _in_vec(line_list, line2[i])))
                    {
                        line_list.push_back(line2[i]);
                    }
                }
            }
        }
        return line_list.size();
    }

    std::pair<std::vector<int>, std::vector<int>> _get_xy(std::vector<int> coords, int params_margin)
    {
        if (coords.empty())
        {
            std::cout << "表格交点数为 0\n";
            return std::pair<std::vector<int>, std::vector<int>>();
        }
        std::vector<int> coord_list;
        int i = 0;
        std::vector<int> sorted_coords(coords.size());
        std::copy(coords.begin(), coords.end(), sorted_coords.begin());
        std::sort(sorted_coords.begin(), sorted_coords.end());
        for (; i < sorted_coords.size() - 1; i++)
        {
            if (sorted_coords[static_cast<std::vector<int, std::allocator<int>>::size_type>(i) + 1] - sorted_coords[i] > params_margin)
            {
                coord_list.push_back(sorted_coords[i]);
            }
        }
        coord_list.push_back(sorted_coords[i]);//包括最后一点，每个点都是取最大的一个
        return std::make_pair(sorted_coords, coord_list);
    }
};

std::vector<std::vector<int>> image_table_recognition::run(std::string img_path, std::string out_path)
{
    srand((unsigned int)time(NULL));

    if (_access(img_path.c_str(), 0) == -1)
    {
        std::cout << "无法打开文件\n";
        return std::vector<std::vector<int>>();
    }

    char buff[250];
    if (GetCurrentDir(buff, 250))
    {
        if (out_path.empty())
        {
            out_path = std::string(buff) + "/" + "output";
        }
        else
        {
            out_path = std::string(buff) + "/" + out_path;
        }
    }
    if (_access(out_path.c_str(), 0) == -1)
    {
        if (_mkdir(out_path.c_str()) == -1)
        {
            std::cout << "文件夹创建失败\n";
            return std::vector<std::vector<int>>();
        }
    }

    //读取图像
    cv::Mat img = cv::imread(img_path);
    if (img.empty())
    {
        std::cout << "图像读取失败\n";
        return std::vector<std::vector<int>>();
    }
    //灰度化处理
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    //二值化处理
    cv::Mat binary;
    cv::adaptiveThreshold(~gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -10);
    cv::imwrite(out_path + "/cell.png", binary);
    //识别横线
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 1));
    cv::Mat erode;
    cv::erode(binary, erode, kernel);//腐蚀
    cv::Mat horizontal_line;
    cv::dilate(erode, horizontal_line, kernel);//膨胀
    cv::imwrite(out_path + "/horizontal_line.png", horizontal_line);
    //识别竖线
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 25));
    cv::erode(binary, erode, kernel);//腐蚀
    cv::Mat vertical_line;
    cv::dilate(erode, vertical_line, kernel);//膨胀
    cv::imwrite(out_path + "/vertical_line.png", vertical_line);
    //识别表格
    cv::Mat table;
    cv::add(horizontal_line, vertical_line, table);
    cv::imwrite(out_path + "/table.png", table);
    //获取表格中所有的线
    auto lines = _where(table);
    std::vector<int> line_ys = lines.first;
    std::vector<int> line_xs = lines.second;
    std::cout << "line_ys size: " << line_ys.size() << "\n";
    auto maxPosition = std::max_element(line_ys.begin(), line_ys.end());
    std::cout << "line_ys max: " << *maxPosition << " at the postion of " << maxPosition - line_ys.begin() << "\n";
    std::cout << "line_xs size: " << line_xs.size() << "\n";
    maxPosition = std::max_element(line_xs.begin(), line_xs.end());
    std::cout << "line_xs max: " << *maxPosition << " at the postion of " << maxPosition - line_xs.begin() << "\n";
    //识别交点
    cv::Mat intersecting_point;
    cv::bitwise_and(horizontal_line, vertical_line, intersecting_point);
    cv::imwrite(out_path + "/intersecting_point.png", intersecting_point);
    //获取表格交点坐标
    auto xys = _where(intersecting_point);
    std::vector<int> ys = xys.first;
    std::vector<int> xs = xys.second;
    std::cout << "ys size: " << ys.size() << "\n";
    maxPosition = std::max_element(ys.begin(), ys.end());
    std::cout << "ys max: " << *maxPosition << " at the postion of " << maxPosition - ys.begin() << "\n";
    std::cout << "xs size: " << xs.size() << "\n";
    maxPosition = std::max_element(xs.begin(), xs.end());
    std::cout << "xs max: " << *maxPosition << " at the postion of " << maxPosition - xs.begin() << "\n";

    auto _ys = _get_xy(ys, params_margin_y);
    std::vector<int> sorted_ys = _ys.first;
    std::vector<int> list_y = _ys.second;
    std::vector<int> list_x = _get_xy(xs, params_margin_x).second;
    std::cout << "sorted_ys size: " << sorted_ys.size() << "\n";
    maxPosition = std::max_element(sorted_ys.begin(), sorted_ys.end());
    std::cout << "sorted_ys max: " << *maxPosition << " at the postion of " << maxPosition - sorted_ys.begin() << "\n";
    std::cout << "list_y size: " << list_y.size() << "\n";
    maxPosition = std::max_element(list_y.begin(), list_y.end());
    std::cout << "list_y max: " << *maxPosition << " at the postion of " << maxPosition - list_y.begin() << "\n";
    std::cout << "list_x size: " << list_x.size() << "\n";
    maxPosition = std::max_element(list_x.begin(), list_x.end());
    std::cout << "list_x max: " << *maxPosition << " at the postion of " << maxPosition - list_x.begin() << "\n";
    std::cout << "横线数：" << list_y.size() << "\n";
    std::cout << "竖线数：" << list_x.size() << "\n";

    //根据交点集获取单元格轮廓并进行过滤
    std::vector<std::vector<int>> coord_list;
    std::map<int, std::vector<std::vector<int>>> coord_map;
    for (int i = 0; i < sorted_ys.size(); i++)
    {
        for (auto y = list_y.begin(); y != list_y.end(); y++)
        {
            for (auto x = list_x.begin(); x != list_x.end(); x++)
            {
                if ((std::abs(*y - ys[i]) < params_dot_margin) and (std::abs(*x - xs[i]) < params_dot_margin) and
                    (not _multi_in_vec(coord_list, *y, *x)))
                {
                    std::vector<int> coord = { *y, *x };
                    coord_list.push_back(coord);
                }
            }
        }
    }
    for (int i = 0; i < list_y.size(); i++)
    {
        std::vector<std::vector<int>> line_list;
        for (auto coord = coord_list.begin(); coord != coord_list.end(); coord++)
        {
            if ((*coord)[0] == list_y[i])
            {
                line_list.push_back(*coord);
            }
        }
        std::sort(line_list.begin(), line_list.end(), [](std::vector<int> v1, std::vector<int> v2) { return v1[1] < v2[1]; });
        std::cout << "888888\n";
        std::cout << line_list.size() << "\n";
        coord_map[i] = line_list;
    }
    std::vector<std::vector<int>> cells_coords;//单元格坐标
    for (int i = 0; i < coord_map.size() - 1; i++)
    {
        std::cout << "777777\n";
        std::cout << coord_map[i].size() << "\n";
        for (int index = 0; index < coord_map[i].size(); index++)
        {
            if (index == coord_map[i].size() - 1)
            {
                break;
            }
            int cell_up = coord_map[i][index][0];//单元格上边框
            int cell_left = coord_map[i][index][1];//单元格左边框
            for (int j = 1; j < coord_map[i].size(); j++)
            {
                int m = i;
                int n = index + j;
                int mark_num = 0;
                if (n == coord_map[i].size())
                {
                    break;
                }
                int cell_down = coord_map[m + 1][0][0];//单元格下边框
                int cell_right = coord_map[i][n][1];//单元格右边框
                while (m <= coord_map.size() - 2)
                {
                    std::vector<int> tmp;
                    for (auto& coord : coord_map[m + 1])
                    {
                        tmp.push_back(coord[1]);
                    }
                    //std::cout
                    //    << _in_vec(tmp, cell_left)
                    //    << _in_vec(tmp, cell_right)
                    //    << " "
                    //    << std::abs(_recognize_line(line_xs, line_xs, line_ys, cell_left, cell_up, cell_down) - (cell_down - cell_up))
                    //    //<= params_line_y
                    //    << " "
                    //    << std::abs(_recognize_line(line_xs, line_xs, line_ys, cell_right, cell_up, cell_down) - (cell_down - cell_up))
                    //    //<= params_line_y
                    //    << " "
                    //    << std::abs(_recognize_line(line_xs, line_ys, line_xs, cell_up, cell_left, cell_right) - (cell_right - cell_left))
                    //    //<= params_line_x
                    //    << " "
                    //    << std::abs(_recognize_line(line_xs, line_ys, line_xs, cell_down, cell_left, cell_right) - (cell_right - cell_left))
                    //    //<= params_line_x
                    //    << "\n";
                    if (_in_vec(tmp, cell_left) and _in_vec(tmp, cell_right) and
                        (std::abs(_recognize_line(line_xs, line_xs, line_ys, cell_left, cell_up, cell_down) - (cell_down - cell_up))
                            <= params_line_y) and
                        (std::abs(_recognize_line(line_xs, line_xs, line_ys, cell_right, cell_up, cell_down) - (cell_down - cell_up))
                            <= params_line_y) and
                        (std::abs(_recognize_line(line_xs, line_ys, line_xs, cell_up, cell_left, cell_right) - (cell_right - cell_left))
                            <= params_line_x) and
                        (std::abs(_recognize_line(line_xs, line_ys, line_xs, cell_down, cell_left, cell_right) - (cell_right - cell_left))
                            <= params_line_x))
                    {
                        std::cout << "666666\n";
                        mark_num = 1;
                        cv::Mat roi1 = img.rowRange(cell_up, cell_down);
                        std::cout << img.size() << " " << img.dims << "\n";
                        std::cout << roi1.size() << " " << roi1.dims << "\n";
                        cv::Mat roi2 = roi1.colRange(cell_left, cell_right);
                        std::vector<int> coords = { cell_up, cell_down, cell_left, cell_right };
                        cells_coords.push_back(coords);

                        int order_num1 = _index(list_y, cell_up);
                        int order_num2 = _index(list_y, cell_down) - 1;
                        int order_num3 = _index(list_y, cell_left);
                        int order_num4 = _index(list_y, cell_right) - 1;

                        std::string img_name = std::to_string(rand());
                        std::string save_dir = out_path + "/" + img_name;
                        if (_access(save_dir.c_str(), 0) == -1)
                        {
                            if (_mkdir(save_dir.c_str()) == -1)
                            {
                                std::cout << "文件夹创建失败\n";
                                continue;
                            }
                        }
                        std::string new_img_name = (img_name + "_" +
                            std::to_string(order_num1) + "_" +
                            std::to_string(order_num2) + "_" +
                            std::to_string(order_num3) + "_" +
                            std::to_string(order_num4) + ".jpg");
                        std::string save_path = save_dir + "/" + new_img_name;
                        cv::imwrite(save_path, roi2);

                        break;
                    }
                    else
                    {
                        m++;
                    }
                }
                if (mark_num == 1)
                {
                    break;
                }
            }
        }
    }
    return cells_coords;
}

int main()
{
    image_table_recognition itr;
    itr.run("qiaofeng.png", "");
    //system("pause");
}
