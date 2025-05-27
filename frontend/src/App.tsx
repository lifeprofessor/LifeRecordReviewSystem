import React, { useState } from 'react';

import {
  ChakraProvider,
  Box,
  VStack,
  Heading,
  Select,
  Textarea,
  Button,
  Text,
  useToast,
  Container,
  Divider,
  Flex,
  useColorModeValue,
  Card,
  CardBody,
  Spinner,
  Badge,
  theme,
  Icon,
  HStack,
} from '@chakra-ui/react';
import { FiFileText, FiEdit, FiBookOpen, FiCheckCircle } from 'react-icons/fi';

const API_URL = 'http://localhost:8000';

// Custom theme extension
const customTheme = {
  ...theme,
  styles: {
    global: {
      body: {
        bg: 'gray.50',
      },
    },
  },
};

function App() {
  const [area, setArea] = useState('');
  const [academicLevel, setAcademicLevel] = useState('');
  const [statement, setStatement] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [evaluation, setEvaluation] = useState('');
  const [feedback, setFeedback] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [example, setExample] = useState('');
  const [isDocumentsLoaded, setIsDocumentsLoaded] = useState(false);
  const [suggestionLength, setSuggestionLength] = useState(0);
  const [sessionId, setSessionId] = useState<string>('');
  const toast = useToast();

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const cardBg = useColorModeValue('white', 'gray.800');
  const headingColor = useColorModeValue('blue.600', 'blue.300');

// 선택된 영역/학업 수준에 따라 문서 로드 API 호출
  const handleLoadDocuments = async () => {
    if (!area || !academicLevel) {
      toast({
        title: '입력 오류',
        description: '활동 영역과 학업 수준을 모두 선택해주세요.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/load-documents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ area, academic_level: academicLevel }),
      });

      if (!response.ok) {
        throw new Error('문서 로드 중 오류가 발생했습니다.');
      }

      const data = await response.json();
      setSessionId(data.session_id);
      setIsDocumentsLoaded(true);
      toast({
        title: '성공',
        description: '문서가 성공적으로 로드되었습니다.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error: unknown) {
      toast({
        title: '오류',
        description: error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  //사용자 문장을 /api/review로 전송, 평가 및 예시문장 결과 수신
  const handleReview = async () => {
    if (!statement) {
      toast({
        title: '입력 오류',
        description: '검토할 문장을 입력해주세요.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!sessionId) {
      toast({
        title: '세션 오류',
        description: '문서를 먼저 로드해주세요.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/review`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          statement,
          session_id: sessionId 
        }),
      });

      if (!response.ok) {
        throw new Error('검토 중 오류가 발생했습니다.');
      }

      const data = await response.json();
      setEvaluation(data.evaluation);
      setFeedback(data.feedback);
      setSuggestion(data.suggestion);
      setSuggestionLength(data.suggestion_length);
      setExample(data.example);
    } catch (error: unknown) {
      toast({
        title: '오류',
        description: error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ChakraProvider theme={customTheme}>
      <Flex minH="100vh" bg="gray.50">
        {/* Sidebar */}
        <Box
          w={{ base: '100%', md: '320px' }}
          minH="100vh"
          bg="#e9f0fa"
          boxShadow="xl"
          p={8}
          display="flex"
          flexDirection="column"
          alignItems="flex-start"
          justifyContent="flex-start"
        >
          <img src="/sdj-logo.png" alt="학교 로고" style={{ width: 196, marginRight: 36 }} />
          <br/>
          <Heading size="md" color="blue.700" mb={8} fontWeight="bold" letterSpacing="tight">
            활동 영역 선택
          </Heading>
          <Text fontSize="sm" color="gray.600" mb={2}>
            검토할 활동 영역을 선택하세요
          </Text>
          <Select
            placeholder="활동 영역 선택"
            value={area}
            onChange={(e) => setArea(e.target.value)}
            size="md"
            bg="white"
            borderColor="gray.300"
            mb={4}
            _hover={{ borderColor: 'blue.400' }}
            _focus={{ borderColor: 'blue.500', boxShadow: '0 0 0 1px blue.500' }}
          >
            <option value="자율/자치활동 특기사항">자율/자치활동 특기사항</option>
            <option value="진로활동 특기사항">진로활동 특기사항</option>
          </Select>
          <Text fontSize="sm" color="gray.600" mb={2}>
            학업수준을 선택하세요
          </Text>
          <Select
            placeholder="학업 수준 선택"
            value={academicLevel}
            onChange={(e) => setAcademicLevel(e.target.value)}
            size="md"
            bg="white"
            borderColor="gray.300"
            mb={6}
            _hover={{ borderColor: 'blue.400' }}
            _focus={{ borderColor: 'blue.500', boxShadow: '0 0 0 1px blue.500' }}
          >
            <option value="상위권">상위권</option>
            <option value="중위권">중위권</option>
            <option value="하위권">하위권</option>
          </Select>
          <Button
            colorScheme="blue"
            w="100%"
            size="lg"
            leftIcon={<Icon as={FiFileText as any} boxSize={5} />}
            mb={2}
            boxShadow="md"
            borderRadius="lg"
            fontWeight="bold"
            onClick={handleLoadDocuments}
            isLoading={isLoading}
            loadingText="로딩 중..."
            _hover={{ bg: 'blue.600', color: 'white', boxShadow: 'lg' }}
          >
            문서 로드
          </Button>
        </Box>

        {/* Main Content */}
        <Box
          flex="1"
          pt={{ base: 2, md: 7 }}
          pb={{ base: 4, md: 12 }}
          px={{ base: 4, md: 12 }}
          maxW="1000px"
          ml={{ base: 0, md: 12 }}
        >
          <VStack spacing={2} align="stretch">
            {/* 시스템명 및 타이틀 */}
            <Box 
              mb={6} 
              p={6} 
              bg="white" 
              borderRadius="xl" 
              boxShadow="lg"
              borderWidth="1px"
              borderColor="gray.100"
            >
              <VStack spacing={4} align="stretch">
                <HStack spacing={3} align="center">
                  <Icon as={FiBookOpen as any} boxSize={8} color="blue.500" />
                  <Heading size="lg" color="blue.800" fontWeight="extrabold">
                  AI 기반 생기부 특기사항 문장 평가 시스템
                  </Heading>
                </HStack>
                
                <Divider borderColor="blue.100" />
                
                <HStack spacing={3} align="center">
                  <Icon as={FiCheckCircle as any} boxSize={6} color="green.500" />
                  <Heading size="md" color="gray.800" fontWeight="bold">
                  특기사항을 입력하면 AI가 적절성 평가 및 개선안을 제시합니다.
                  </Heading>
                </HStack>
                
                <Text 
                  color="gray.600" 
                  fontSize="md"
                  bg="gray.50"
                  p={4}
                  borderRadius="md"
                  borderLeft="4px solid"
                  borderColor="blue.400"
                >
                  이 시스템은 문장의 적절성, 표현력, 구체성을 분석해 개선 방향을 안내합니다.
                </Text>
              </VStack>
            </Box>

            {/* 문서가 로드된 경우에만 검토 카드 표시 */}
            {isDocumentsLoaded && (
              <Card
                bg={cardBg}
                boxShadow="2xl"
                borderRadius="2xl"
                borderWidth="1px"
                borderColor={borderColor}
                p={8}
              >
                <CardBody>
                  <VStack spacing={8} align="stretch">
                    <Box>
                      <Text mb={2} fontWeight="semibold" color="gray.700" fontSize="2xl">
                        검토할 문장
                      </Text>
                      <Textarea
                        placeholder="검토할 문장을 입력하세요..."
                        value={statement}
                        onChange={(e) => setStatement(e.target.value)}
                        minHeight="120px"
                        size="lg"
                        bg="white"
                        borderColor="gray.300"
                        fontSize="md"
                        _hover={{ borderColor: 'blue.400' }}
                        _focus={{ borderColor: 'blue.500', boxShadow: '0 0 0 1px blue.500' }}
                      />
                    </Box>
                    <Button
                      colorScheme="green"
                      size="lg"
                      leftIcon={<Icon as={FiEdit as any} boxSize={5} />}
                      onClick={handleReview}
                      isLoading={isLoading}
                      loadingText="검토 중..."
                      fontWeight="bold"
                      borderRadius="lg"
                      boxShadow="md"
                      _hover={{ bg: 'green.600', color: 'white', boxShadow: 'lg' }}
                    >
                      검토하기
                    </Button>

                    {evaluation && (
                      <Box mt={4}>
                        <Flex align="center" mb={4}>
                          <Badge colorScheme="blue" fontSize="md" px={3} py={1} borderRadius="full">
                            1️⃣ 적합성 평가
                          </Badge>
                        </Flex>
                        <Text
                          whiteSpace="pre-line"
                          lineHeight={1.9}
                          fontSize="lg"
                          bg="gray.50"
                          p={4}
                          borderRadius="md"
                        >
                          {evaluation}
                        </Text>
                        <Divider my={6} />
                        <Flex align="center" mb={4}>
                          <Badge colorScheme="purple" fontSize="md" px={3} py={1} borderRadius="full">
                            2️⃣ 검토 의견
                          </Badge>
                        </Flex>
                        <Text
                          whiteSpace="pre-line"
                          lineHeight={1.9}
                          fontSize="lg"
                          bg="gray.50"
                          p={4}
                          borderRadius="md"
                        >
                          {feedback}
                        </Text>
                        <Divider my={6} />
                        <Flex align="center" mb={4}>
                          <Badge colorScheme="green" fontSize="md" px={3} py={1} borderRadius="full">
                            3️⃣ 개선 제안
                          </Badge>
                          <Text ml={2} fontSize="sm" color="gray.500">
                            ({suggestionLength}자)
                          </Text>
                        </Flex>
                        <Text
                          whiteSpace="pre-line"
                          bg="gray.50"
                          p={4}
                          borderRadius="md"
                        >
                          {suggestion}
                        </Text>
                      </Box>
                    )}
                  </VStack>
                </CardBody>
              </Card>
            )}
          </VStack>
        </Box>
      </Flex>
    </ChakraProvider>
  );
}

export default App; 