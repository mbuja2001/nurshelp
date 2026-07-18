#!/usr/bin/env node

require('dotenv').config();
const mongoose = require('mongoose');

const mongodb_uri = process.env.MONGODB_URI || 'mongodb://localhost:27017/nursehelpdesk';

mongoose.connect(mongodb_uri).then(async () => {
  const db = mongoose.connection.getClient().db('nursehelpdesk');
  const result = await db.collection('clinicalStaff').updateOne(
    { name: 'Lisa Anderson' },
    { $set: { specialty: 'Cardiology' } }
  );
  console.log('Updated:', result.modifiedCount, 'documents');
  
  const doc = await db.collection('clinicalStaff').findOne({ name: 'Lisa Anderson' });
  console.log('Updated document:', { name: doc.name, specialty: doc.specialty });
  
  process.exit(0);
}).catch(e => {
  console.error('Error:', e.message);
  process.exit(1);
});
