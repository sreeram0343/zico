export type SegmentType = 'FLIGHT' | 'HOTEL' | 'ACTIVITY' | 'TRANSFER';

export interface Location {
  name: string;
  iata_code?: string;
  lat?: number;
  lng?: number;
}

export interface TripSegment {
  id: string;
  type: SegmentType;
  title: string;
  start_time: string;
  end_time: string;
  location: Location;
  cost: number;
  currency: string;
  metadata?: Record<string, any>;
  is_confirmed: boolean;
}

export interface PendingAction {
  action_id: string;
  action_type: 'BOOKING' | 'CANCELLATION' | 'PAYMENT' | 'RESCHEDULE';
  description: string;
  payload: Record<string, any>;
  requires_explicit_approval: boolean;
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
}

export interface InterruptEvent {
  action_id: string;
  action_type: string;
  description: string;
  payload: Record<string, any>;
  prompt: string;
  requires_explicit_approval: boolean;
}

export interface SourceCitation {
  title: string;
  url: string;
  score?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  text: string;
  node?: string;
  time: string;
  sources?: SourceCitation[];
}

export interface StreamEvent {
  type:
    | 'node_update'
    | 'interrupt'
    | 'voice_chunk'
    | 'transcript'
    | 'turn_complete'
    | 'error'
    | 'status'
    | 'tool_call'
    | 'token'
    | 'state_update';
  node?: string;
  output?: Record<string, any>;
  message?: string;
  content?: string;
  interrupt_value?: InterruptEvent;
  prompt?: string;
  audio_base64?: string;
  text?: string;
  trip_id?: string;
  tool?: string;
  input?: any;
  itinerary?: TripSegment[];
  sources?: SourceCitation[];
}

export interface FlightOption {
  id?: string;
  airline: string;
  flightNumber?: string;
  departure: string;
  departureAirport: string;
  arrival: string;
  arrivalAirport: string;
  duration: string;
  stops?: string;
  price?: string;
  currency?: string;
  aircraft?: string;
  baggage?: string;
  cabin?: string;
}
