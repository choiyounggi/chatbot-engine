package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.AnalysisResult;
import com.yk.chatbot.lasa.Solve;
import com.yk.chatbot.lasa.SolutionResult;
import com.yk.chatbot.service.WeatherService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;

/**
 * 간단한 문제 해결기 구현체
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class SimpleSolver implements Solve {

    private final WeatherService weatherService;
    
    private static final Map<String, String> RESPONSE_TEMPLATES = Map.of(
        "greeting", "안녕하세요! 무엇을 도와드릴까요?",
        "weather", "%s의 현재 날씨는 %s입니다.",
        "temperature", "%s의 현재 기온은 %d°C입니다.",
        "time", "현재 시간은 %s입니다.",
        "bye", "안녕히 가세요! 좋은 하루 되세요!",
        "thanks", "천만에요! 더 필요한 것이 있으면 말씀해주세요.",
        "help", "저는 날씨, 시간 정보를 알려드리거나 간단한 대화가 가능해요. '서울 날씨 어때?'나 '지금 몇시야?' 같은 질문을 해보세요.",
        "error", "죄송합니다. 이해하지 못했습니다. 다른 방식으로 말씀해주시겠어요?",
        "fallback", "잘 이해하지 못했습니다. 도움이 필요하시면 '도움말'이라고 입력해주세요."
    );
    
    @Override
    public SolutionResult solve(AnalysisResult result) {
        if (result == null || result.getIntent() == null) {
            return createErrorResult();
        }
        
        String intent = result.getIntent();
        Map<String, String> entities = result.getEntities();
        
        log.info("의도 처리 중: {}, 엔티티: {}", intent, entities);
        
        try {
            // 의도에 따른 처리 로직
            switch (intent) {
                case "greeting":
                    return createSuccessResult(intent);
                    
                case "weather":
                    String location = entities.getOrDefault("location", "서울");
                    String weather = weatherService.getWeatherForLocation(location);
                    return createSuccessResult(intent)
                            .addData("location", location)
                            .addData("weather", weather);
                    
                case "temperature":
                    location = entities.getOrDefault("location", "서울");
                    int temp = weatherService.getTemperatureForLocation(location);
                    return createSuccessResult(intent)
                            .addData("location", location)
                            .addData("temperature", temp);
                    
                case "time":
                    String currentTime = LocalDateTime.now()
                            .format(DateTimeFormatter.ofPattern("a h시 mm분"));
                    return createSuccessResult(intent)
                            .addData("time", currentTime);
                    
                case "bye":
                case "thanks":
                case "help":
                    return createSuccessResult(intent);
                    
                case "error":
                    return createErrorResult();
                    
                default:
                    return createFallbackResult();
            }
        } catch (Exception e) {
            log.error("의도 처리 중 오류 발생", e);
            return createErrorResult();
        }
    }
    
    private SolutionResult createSuccessResult(String intent) {
        return SolutionResult.builder()
                .status("success")
                .originalIntent(intent)
                .responseTemplate(RESPONSE_TEMPLATES.getOrDefault(intent, RESPONSE_TEMPLATES.get("fallback")))
                .build();
    }
    
    private SolutionResult createErrorResult() {
        return SolutionResult.builder()
                .status("error")
                .originalIntent("error")
                .responseTemplate(RESPONSE_TEMPLATES.get("error"))
                .build();
    }
    
    private SolutionResult createFallbackResult() {
        return SolutionResult.builder()
                .status("fallback")
                .originalIntent("fallback")
                .responseTemplate(RESPONSE_TEMPLATES.get("fallback"))
                .build();
    }
}
