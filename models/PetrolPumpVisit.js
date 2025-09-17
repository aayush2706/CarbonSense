const mongoose = require('mongoose');

const visitEntrySchema = new mongoose.Schema({
  visitDate: { type: Date, required: true, default: Date.now },
  petrolPumpName: { type: String, trim: true },
  petrolPumpLocation: {
    address: String,
    city: String,
    state: String
  },
  fuelType: { type: String, required: true, enum: ['Petrol', 'Diesel', 'CNG'] },
  litresFilled: { type: Number, required: true, min: 0.01 },
  pricePaid: { type: Number, required: true, min: 0 },
  pricePerLitre: { type: Number, required: true, min: 0 },
  paymentMethod: { type: String, enum: ['Cash', 'Card', 'UPI', 'Wallet'], default: 'Cash' },
  odometerReading: { type: Number, min: 0 }
}, { _id: false });  

const petrolPumpVisitSchema = new mongoose.Schema({
  registrationNumber: { type: String, required: true, uppercase: true, trim: true, index: true },
  visits: [visitEntrySchema]

}, {
  timestamps: true
});

petrolPumpVisitSchema.index({ registrationNumber: 1 });
module.exports = mongoose.model('PetrolPumpVisit', petrolPumpVisitSchema);
