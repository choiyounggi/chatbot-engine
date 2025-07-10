package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.dto.ChatResponse;
import com.yk.chatbot.lasa.Answer;
import com.yk.chatbot.lasa.SolutionResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.ZoneId;

/**
 * 간단한 응답 생성기 구현체
 */
@Slf4j
@Component
public class SimpleAnswerer implements Answer {

    @Override
    public ChatResponse answer(SolutionResult result) {
        if (result == null) {
            return createErrorResponse("시스템 오류가 발생했습니다.");
        }
        
        try {
            String responseMessage = formatResponseMessage(result);
            
            return ChatResponse.builder()
                    .message(responseMessage)
                    .intent(result.getOriginalIntent())
                    .entities(result.getData())
                    .confidence(result.isSuccess() ? 1.0 : 0.5)
                    .timestamp(System.currentTimeMillis())
                    .build();
        } catch (Exception e) {
            log.error("응답 생성 중 오류 발생", e);
            return createErrorResponse("응답을 생성하는 중 오류가 발생했습니다.");
        }
    }
    
    /**
     * 응답 메시지 형식화
     * 
     * @param result 해결 결과
     * @return 형식화된 메시지 문자열
     */
    private String formatResponseMessage(SolutionResult result) {
        String template = result.getResponseTemplate();
        
        // 템플릿이 없는 경우 기본 응답
        if (template == null) {
            return "죄송합니다. 응답을 생성할 수 없습니다.";
        }
        
        // 특수 응답 형식 처리
        switch (result.getOriginalIntent()) {
            case "greeting":
                // 랜덤 인사말 사용
                return String.format(template, 
                        result.getData().getOrDefault("greeting", "안녕하세요! 무엇을 도와드릴까요?"));

            case "weather":
                // 위치, 날씨 상태, 상세 설명 포함
                String location = String.valueOf(result.getData().getOrDefault("location", "서울"));
                String weather = String.valueOf(result.getData().getOrDefault("weather", "알 수 없음"));
                String weatherDetail = String.valueOf(result.getData().getOrDefault("weatherDetail", ""));

                // 날짜/시간 정보 추가
                String dateTime = String.valueOf(result.getData().getOrDefault("dateTime", ""));
                if (!dateTime.isEmpty()) {
                    dateTime = " (" + dateTime + " 기준)";
                }

                return String.format(template, location, weather, weatherDetail) + dateTime;

            case "temperature":
                location = String.valueOf(result.getData().getOrDefault("location", "서울"));
                String tempValue = String.valueOf(result.getData().getOrDefault("temperature", "0"));
                int temperature = Integer.parseInt(tempValue);
                String tempDescription = String.valueOf(result.getData().getOrDefault("tempDescription", ""));

                return String.format(template, location, temperature, tempDescription);
            case "time":
                return String.format(template, 
                        result.getData().getOrDefault("time", 
                                LocalDateTime.now(ZoneId.of("Asia/Seoul"))
                                        .toString()));
                
            case "bye":
                // 랜덤 작별 인사 사용
                return String.format(template, 
                        result.getData().getOrDefault("bye", "안녕히 가세요! 좋은 하루 되세요!"));
                
            default:
                // 일반 템플릿은 그대로 반환
                return template;
        }
    }
    
    /**
     * 오류 응답 생성
     * 
     * @param errorMessage 오류 메시지
     * @return 오류 응답
     */
    private ChatResponse createErrorResponse(String errorMessage) {
        return ChatResponse.builder()
                .message(errorMessage)
                .intent("error")
                .confidence(0.0)
                .timestamp(System.currentTimeMillis())
                .build();
    }
}
