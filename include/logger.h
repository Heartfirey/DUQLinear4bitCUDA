#ifndef LOGGER_H_
#define LOGGER_H_

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>
#include <spdlog/fmt/ostr.h>
#include <iostream>
#include <memory>
#include <string>

#define LOG_INFO(...) SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), __VA_ARGS__);SPDLOG_LOGGER_INFO(spdlog::get("daily_logger"), __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__);SPDLOG_LOGGER_WARN(spdlog::get("daily_logger"), __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__);SPDLOG_LOGGER_ERROR(spdlog::get("daily_logger"), __VA_ARGS__)

extern void LoggerInit();

extern void LoggerDrop();

namespace AnsiColors{
    // Formatting codes
    constexpr std::string_view reset = "\033[m";
    constexpr std::string_view bold = "\033[1m";
    constexpr std::string_view dark = "\033[2m";
    constexpr std::string_view underline = "\033[4m";
    constexpr std::string_view blink = "\033[5m";
    constexpr std::string_view reverse = "\033[7m";
    constexpr std::string_view concealed = "\033[8m";
    constexpr std::string_view clear_line = "\033[K";

    // Foreground colors
    constexpr std::string_view black = "\033[30m";
    constexpr std::string_view red = "\033[31m";
    constexpr std::string_view green = "\033[32m";
    constexpr std::string_view yellow = "\033[33m";
    constexpr std::string_view blue = "\033[34m";
    constexpr std::string_view magenta = "\033[35m";
    constexpr std::string_view cyan = "\033[36m";
    constexpr std::string_view white = "\033[37m";

    /// Background colors
    constexpr std::string_view on_black = "\033[40m";
    constexpr std::string_view on_red = "\033[41m";
    constexpr std::string_view on_green = "\033[42m";
    constexpr std::string_view on_yellow = "\033[43m";
    constexpr std::string_view on_blue = "\033[44m";
    constexpr std::string_view on_magenta = "\033[45m";
    constexpr std::string_view on_cyan = "\033[46m";
    constexpr std::string_view on_white = "\033[47m";

    /// Bold colors
    constexpr std::string_view yellow_bold = "\033[33m\033[1m";
    constexpr std::string_view red_bold = "\033[31m\033[1m";
    constexpr std::string_view bold_on_red = "\033[1m\033[41m";
}




#endif /*LOGGER_H_ */
