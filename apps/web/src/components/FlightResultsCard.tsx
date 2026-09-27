'use client';

import React, { useState } from 'react';
import { FlightOption } from '@/types';
import { ArrowRight, Info, X, Shield, Luggage, Clock, CheckCircle } from 'lucide-react';

interface FlightResultsCardProps {
  flights?: FlightOption[];
  origin?: string;
  destination?: string;
  date?: string;
}

const DEFAULT_DEMO_FLIGHTS: FlightOption[] = [
  {
    id: 'f1',
    airline: 'Emirates',
    flightNumber: 'EK-523',
    departure: '10:35',
    departureAirport: 'COK',
    arrival: '12:55',
    arrivalAirport: 'DXB',
    duration: '4h 20m',
    stops: 'Non-stop',
    price: '₹ 18,450',
    aircraft: 'Boeing 777-300ER',
    baggage: '30 kg check-in, 7 kg cabin',
    cabin: 'Economy',
  },
  {
    id: 'f2',
    airline: 'IndiGo',
    flightNumber: '6E-1451',
    departure: '12:10',
    departureAirport: 'COK',
    arrival: '16:35',
    arrivalAirport: 'DXB',
    duration: '4h 25m',
    stops: 'Non-stop',
    price: '₹ 16,900',
    aircraft: 'Airbus A321neo',
    baggage: '30 kg check-in, 7 kg cabin',
    cabin: 'Economy',
  },
  {
    id: 'f3',
    airline: 'Air India',
    flightNumber: 'AI-933',
    departure: '14:20',
    departureAirport: 'COK',
    arrival: '19:00',
    arrivalAirport: 'DXB',
    duration: '4h 40m',
    stops: 'Non-stop',
    price: '₹ 17,850',
    aircraft: 'Boeing 787-8 Dreamliner',
    baggage: '25 kg check-in, 7 kg cabin',
    cabin: 'Economy',
  },
  {
    id: 'f4',
    airline: 'flydubai',
    flightNumber: 'FZ-454',
    departure: '18:50',
    departureAirport: 'COK',
    arrival: '23:20',
    arrivalAirport: 'DXB',
    duration: '4h 30m',
    stops: 'Non-stop',
    price: '₹ 15,600',
    aircraft: 'Boeing 737 MAX 8',
    baggage: '20 kg check-in, 7 kg cabin',
    cabin: 'Economy',
  },
];

function getAirlineBadge(airline: string) {
  const lower = airline.toLowerCase();
  if (lower.includes('emirates')) {
    return (
      <div className="w-8 h-8 rounded-lg bg-[#D71920]/10 border border-[#D71920]/20 flex items-center justify-center font-bold text-xs text-[#D71920] shrink-0">
        EK
      </div>
    );
  }
  if (lower.includes('indigo')) {
    return (
      <div className="w-8 h-8 rounded-lg bg-[#001B94] flex items-center justify-center font-bold text-xs text-white shrink-0">
        6E
      </div>
    );
  }
  if (lower.includes('air india')) {
    return (
      <div className="w-8 h-8 rounded-lg bg-[#E31E24]/10 border border-[#E31E24]/20 flex items-center justify-center font-bold text-xs text-[#E31E24] shrink-0">
        AI
      </div>
    );
  }
  return (
    <div className="w-8 h-8 rounded-lg bg-[#F5BE22]/15 border border-[#F5BE22]/30 flex items-center justify-center font-bold text-xs text-[#101828] shrink-0">
      FZ
    </div>
  );
}

export function FlightResultsCard({ flights }: FlightResultsCardProps) {
  const displayFlights = flights && flights.length > 0 ? flights : DEFAULT_DEMO_FLIGHTS;
  const [selectedFlight, setSelectedFlight] = useState<FlightOption | null>(null);

  return (
    <div className="mt-4 bg-white rounded-2xl border border-[#E5E7EB] p-3 sm:p-4 shadow-sm w-full">
      {/* Flight Rows */}
      <div className="divide-y divide-[#F2F4F7]">
        {displayFlights.map((flight, idx) => (
          <div
            key={flight.id || `flight-${idx}`}
            className="py-3 px-1 sm:px-2 flex flex-col md:flex-row md:items-center justify-between gap-3 hover:bg-[#FFFDF7] rounded-xl transition-colors"
          >
            {/* Airline */}
            <div className="flex items-center gap-3 min-w-[130px]">
              {getAirlineBadge(flight.airline)}
              <div>
                <span className="font-bold text-sm text-[#101828] block">
                  {flight.airline}
                </span>
                {flight.flightNumber && (
                  <span className="text-[11px] text-[#667085] font-mono">
                    {flight.flightNumber}
                  </span>
                )}
              </div>
            </div>

            {/* Flight Timing & Route */}
            <div className="flex items-center gap-4 sm:gap-6 justify-between md:justify-center flex-1">
              {/* Departure */}
              <div className="text-left md:text-right min-w-[65px]">
                <span className="text-base font-extrabold text-[#101828] block">
                  {flight.departure}
                </span>
                <span className="text-xs font-semibold text-[#667085]">
                  {flight.departureAirport}
                </span>
              </div>

              {/* Route Indicator */}
              <div className="flex flex-col items-center px-2">
                <span className="text-[11px] text-[#667085] font-medium whitespace-nowrap">
                  {flight.duration}
                </span>
                <div className="flex items-center gap-1 my-0.5 text-[#F5BE22]">
                  <div className="h-[2px] w-6 sm:w-10 bg-[#FDE266]" />
                  <ArrowRight className="w-3.5 h-3.5 text-[#F5BE22]" />
                </div>
                <span className="text-[10px] text-[#12B76A] font-semibold">
                  {flight.stops || 'Non-stop'}
                </span>
              </div>

              {/* Arrival */}
              <div className="text-right min-w-[65px]">
                <span className="text-base font-extrabold text-[#101828] block">
                  {flight.arrival}
                </span>
                <span className="text-xs font-semibold text-[#667085]">
                  {flight.arrivalAirport}
                </span>
              </div>
            </div>

            {/* Price & Action */}
            <div className="flex items-center justify-between md:justify-end gap-3 sm:gap-4 shrink-0 border-t md:border-t-0 pt-2 md:pt-0 border-[#F2F4F7]">
              <span className="text-base sm:text-lg font-extrabold text-[#027A48]">
                {flight.price}
              </span>
              <button
                onClick={() => setSelectedFlight(flight)}
                className="px-3.5 py-1.5 rounded-full bg-[#FFF6D8] hover:bg-[#F7C948] text-[#101828] text-xs font-bold transition-all duration-150 shadow-sm active:scale-95 focus:outline-none focus:ring-2 focus:ring-[#F7C948]"
                aria-label={`View details for ${flight.airline} flight`}
              >
                View Details
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Flight Data Disclaimer */}
      <div className="mt-3 pt-3 border-t border-[#F2F4F7] flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-[11px] text-[#667085]">
        <div className="flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-[#D99E10] shrink-0" />
          <span>Prices are approximate and may change. Please check with the airline for the latest details.</span>
        </div>
        <span className="text-[#98A2B3] shrink-0 font-medium">10:24 AM</span>
      </div>

      {/* Modal Dialog for View Details */}
      {selectedFlight && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-in fade-in duration-150"
          role="dialog"
          aria-modal="true"
          aria-labelledby="flight-modal-title"
        >
          <div className="bg-white rounded-3xl max-w-md w-full p-6 shadow-2xl border border-[#E5E7EB] relative animate-in zoom-in-95 duration-150">
            <button
              onClick={() => setSelectedFlight(null)}
              className="absolute top-4 right-4 p-1.5 rounded-full text-[#667085] hover:bg-slate-100 transition-colors"
              aria-label="Close details"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 mb-5">
              {getAirlineBadge(selectedFlight.airline)}
              <div>
                <h3 id="flight-modal-title" className="text-lg font-bold text-[#101828]">
                  {selectedFlight.airline} {selectedFlight.flightNumber}
                </h3>
                <p className="text-xs text-[#667085]">
                  {selectedFlight.cabin || 'Economy'} Class • {selectedFlight.aircraft || 'Commercial Jetliner'}
                </p>
              </div>
            </div>

            <div className="bg-[#FFFDF7] rounded-2xl p-4 border border-[#F2F4F7] space-y-3 mb-5">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xl font-extrabold text-[#101828]">{selectedFlight.departure}</span>
                  <span className="text-xs block text-[#667085] font-semibold">{selectedFlight.departureAirport}</span>
                </div>
                <div className="text-center">
                  <span className="text-xs text-[#667085] font-medium">{selectedFlight.duration}</span>
                  <div className="flex items-center gap-1 my-0.5 text-[#F5BE22]">
                    <div className="h-[2px] w-8 bg-[#FDE266]" />
                    <ArrowRight className="w-3 h-3 text-[#F5BE22]" />
                  </div>
                  <span className="text-[10px] text-[#027A48] font-bold">{selectedFlight.stops}</span>
                </div>
                <div className="text-right">
                  <span className="text-xl font-extrabold text-[#101828]">{selectedFlight.arrival}</span>
                  <span className="text-xs block text-[#667085] font-semibold">{selectedFlight.arrivalAirport}</span>
                </div>
              </div>
            </div>

            <div className="space-y-2.5 text-xs text-[#344054] mb-6">
              <div className="flex items-center gap-2.5">
                <Luggage className="w-4 h-4 text-[#D99E10]" />
                <span>Baggage: <strong>{selectedFlight.baggage}</strong></span>
              </div>
              <div className="flex items-center gap-2.5">
                <Clock className="w-4 h-4 text-[#D99E10]" />
                <span>On-time reliability rating: <strong>94% historical performance</strong></span>
              </div>
              <div className="flex items-center gap-2.5">
                <Shield className="w-4 h-4 text-[#D99E10]" />
                <span>Ticket conditions: Free cancellation up to 24h prior to departure</span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-[#F2F4F7]">
              <div>
                <span className="text-xs text-[#667085]">Total estimated fare</span>
                <p className="text-xl font-black text-[#027A48]">{selectedFlight.price}</p>
              </div>
              <button
                onClick={() => setSelectedFlight(null)}
                className="px-5 py-2.5 rounded-full bg-[#F7C948] hover:bg-[#F5BE22] font-bold text-xs text-[#101828] shadow-sm transition-colors"
              >
                Close Details
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
