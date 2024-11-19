#include <logger.h>

void LoggerInit()
{
	auto logger = spdlog::rotating_logger_mt("daily_logger", "logs/log.txt", 1024 * 1024 * 20,  5);

	// auto logger = spdlog::daily_logger_mt("daily_logger", "LogFolder/logs/daily.txt", 2, 30);

	logger->flush_on(spdlog::level::warn);

	spdlog::flush_every(std::chrono::seconds(3));

	auto console = spdlog::stdout_color_mt("console");
	spdlog::set_default_logger(console);

	spdlog::set_level(spdlog::level::info);

	spdlog::set_pattern("%^[%Y-%m-%d %H:%M:%S][%l][%s:%#]%$ %v");
}


void LoggerDrop()
{
	spdlog::drop_all();
}
