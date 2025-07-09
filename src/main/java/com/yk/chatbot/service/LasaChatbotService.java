package com.yk.chatbot.service;

import com.yk.chatbot.dto.ChatRequest;
import com.yk.chatbot.dto.ChatResponse;
import com.yk.chatbot.lasa.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

/**
 * LASA 프레임워크를 적용한 챗봇 서비스 클래스입니다.
 * Listen-Analyze-Solve-Answer 패턴을 구현합니다.
 */
@Service
@RequiredArgsConstructor
public class LasaChatbotService {

    private final Listen listener;
    private final Analyze analyzer;
    private final Solve solver;
    private final Answer answerer;

    /**
     * 사용자 입력에 대한 응답을 LASA 프레임워크를 통해 생성합니다.
     * @param request 사용자 요청
     * @return 챗봇의 응답
     */
    public ChatResponse process(ChatRequest request) {
        try {
            // 1. Listen: 사용자 입력 수신 및 전처리
            String preprocessedMessage = listener.listen(request);
            
            // 2. Analyze: 의도 및 엔티티 추출
            AnalysisResult analysisResult = analyzer.analyze(preprocessedMessage);
            
            // 3. Solve: 문제 해결 및 응답 전략 수립
            SolutionResult solutionResult = solver.solve(analysisResult);
            
            // 4. Answer: 최종 응답 생성
            return answerer.answer(solutionResult);
        } catch (Exception e) {
            return ChatResponse.builder()
                    .reply("처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                    .intent("error")
                    .build();
        }
    }
}
