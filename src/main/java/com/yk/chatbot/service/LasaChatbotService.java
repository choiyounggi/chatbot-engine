package com.yk.chatbot.service;

import com.yk.chatbot.dto.ChatRequest;
import com.yk.chatbot.dto.ChatResponse;
import com.yk.chatbot.lasa.Analyze;
import com.yk.chatbot.lasa.AnalysisResult;
import com.yk.chatbot.lasa.Answer;
import com.yk.chatbot.lasa.Listen;
import com.yk.chatbot.lasa.Solve;
import com.yk.chatbot.lasa.SolutionResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * LASA 프레임워크 기반 챗봇 서비스
 * Listen, Analyze, Solve, Answer 단계를 통해 사용자 요청을 처리합니다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class LasaChatbotService {

    private final Listen listener;
    private final Analyze analyzer;
    private final Solve solver;
    private final Answer answerer;
    
    /**
     * 사용자 메시지를 처리하고 응답을 생성합니다.
     * 
     * @param request 사용자 요청
     * @return 챗봇 응답
     */
    public ChatResponse process(ChatRequest request) {
        if (request == null || request.getMessage() == null || request.getMessage().trim().isEmpty()) {
            return createErrorResponse("유효하지 않은 요청입니다.");
        }
        
        try {
            log.info("사용자 메시지 처리 시작: {}", request.getMessage());
            
            // LASA 단계 1: Listen - 사용자 입력 전처리
            String preprocessedMessage = listener.listen(request);
            log.debug("전처리된 메시지: {}", preprocessedMessage);
            
            // LASA 단계 2: Analyze - 의도 및 엔티티 분석
            AnalysisResult analysisResult = analyzer.analyze(preprocessedMessage, request.getUserId());
            log.debug("분석 결과: intent={}, confidence={}, entities={}", 
                    analysisResult.getIntent(), 
                    analysisResult.getConfidence(), 
                    analysisResult.getEntities());
            
            // LASA 단계 3: Solve - 문제 해결 및 응답 준비
            SolutionResult solutionResult = solver.solve(analysisResult);
            log.debug("해결 결과: status={}, intent={}", 
                    solutionResult.getStatus(), 
                    solutionResult.getOriginalIntent());
            
            // LASA 단계 4: Answer - 최종 응답 생성
            ChatResponse response = answerer.answer(solutionResult);
            log.info("응답 생성 완료: {}", response.getMessage());
            
            return response;
        } catch (Exception e) {
            log.error("메시지 처리 중 오류 발생", e);
            return createErrorResponse("처리 중 오류가 발생했습니다: " + e.getMessage());
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
