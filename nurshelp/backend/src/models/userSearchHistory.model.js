import mongoose from 'mongoose';

const userSearchHistorySchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      index: true,
      refPath: 'userModel'
    },
    userModel: {
      type: String,
      required: true,
      enum: ['ClinicalStaff', 'Manager', 'Practitioner'],
      index: true
    },
    term: {
      type: String,
      required: true,
      trim: true,
      index: true
    }
  },
  {
    timestamps: true
  }
);

userSearchHistorySchema.index({ userId: 1, term: 1 }, { unique: true });
userSearchHistorySchema.index({ userId: 1, updatedAt: -1 });

export const UserSearchHistory = mongoose.model('UserSearchHistory', userSearchHistorySchema);
