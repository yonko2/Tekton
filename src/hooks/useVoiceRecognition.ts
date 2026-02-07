import { useRef, useState, useCallback, useEffect } from 'react';
import { useSandbox } from '@/context/SandboxContext';
import { parseShapeType, parseColorName, getColorHex, getRandomColor } from '@/constants/shapes';
import type { VoiceCommand, Vector3Tuple } from '@/types';


type SpeechRecognitionType = typeof window extends { SpeechRecognition: infer T } ? T : unknown;

function getSpeechRecognition(): SpeechRecognitionType | null {
  if (typeof window === 'undefined') return null;
  
  return (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition ?? null;
}

export function useVoiceRecognition() {
  const { state, addObject, removeObject, clearAllObjects, setVoice } = useSandbox();

  const recognitionRef = useRef<InstanceType<ReturnType<typeof getSpeechRecognition> & (new () => unknown)> | null>(null);
  const [isListening, setIsListening] = useState(false);
  const isSupported = getSpeechRecognition() !== null;

  
  const pointerPosRef = useRef<Vector3Tuple>([0, 0.5, 0]);
  useEffect(() => {
    if (state.gesture.pointerPosition) {
      pointerPosRef.current = state.gesture.pointerPosition;
    }
  }, [state.gesture.pointerPosition]);

  
  const selectedIdRef = useRef<string | null>(null);
  useEffect(() => {
    selectedIdRef.current = state.selectedObjectId;
  }, [state.selectedObjectId]);

  
  const parseCommand = useCallback((transcript: string): VoiceCommand | null => {
    const lower = transcript.toLowerCase().trim();

    if (lower.includes('clear all') || lower.includes('clear everything')) {
      return { type: 'clear' };
    }
    if (lower.includes('delete') || lower.includes('remove')) {
      return { type: 'delete' };
    }
    if (lower.includes('create') || lower.includes('make') || lower.includes('add') || lower.includes('spawn')) {
      const shape = parseShapeType(lower);
      const color = parseColorName(lower);
      return { type: 'create', shape: shape ?? 'cube', color: color ?? undefined };
    }
    return null;
  }, []);

  
  const executeCommand = useCallback(
    (cmd: VoiceCommand) => {
      switch (cmd.type) {
        case 'create': {
          const pos: Vector3Tuple = [...pointerPosRef.current];
          
          if (pos[1] < 0.5) pos[1] = 0.5;
          const colorHex = cmd.color ? getColorHex(cmd.color) : getRandomColor();
          addObject(cmd.shape ?? 'cube', pos, colorHex);
          break;
        }
        case 'delete': {
          if (selectedIdRef.current) {
            removeObject(selectedIdRef.current);
          }
          break;
        }
        case 'clear':
          clearAllObjects();
          break;
      }
    },
    [addObject, removeObject, clearAllObjects],
  );

  
  const startListening = useCallback(() => {
    const Ctor = getSpeechRecognition();
    if (!Ctor) {
      setVoice({ error: 'Speech recognition not supported in this browser' });
      return;
    }

    
    const recognition = new (Ctor as any)();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event: { results: { length: number; [key: number]: { [key: number]: { transcript: string } } } }) => {
      const last = event.results[event.results.length - 1];
      const transcript = last[0].transcript;
      setVoice({ lastCommand: transcript });

      const cmd = parseCommand(transcript);
      if (cmd) {
        setVoice({ lastParsedCommand: cmd });
        executeCommand(cmd);
      }
    };

    recognition.onerror = (event: { error: string }) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      console.warn('Speech recognition error:', event.error);
      setVoice({ error: event.error });
    };

    recognition.onend = () => {
      
      if (recognitionRef.current) {
        try {
          recognition.start();
        } catch {
          
        }
      }
    };

    recognition.start();
    recognitionRef.current = recognition;
    setIsListening(true);
    setVoice({ isListening: true, error: null });
  }, [parseCommand, executeCommand, setVoice]);

  
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      const r = recognitionRef.current;
      recognitionRef.current = null;
      
      (r as any).stop?.();
    }
    setIsListening(false);
    setVoice({ isListening: false });
  }, [setVoice]);

  
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        
        (recognitionRef.current as any).stop?.();
        recognitionRef.current = null;
      }
    };
  }, []);

  return { isSupported, isListening, startListening, stopListening };
}
