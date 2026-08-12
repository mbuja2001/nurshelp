export const enrichSelectedBedWithWard = (selectedBed, wardInfo = {}) => {
  if (!selectedBed) return null;

  const wardId = selectedBed.wardId
    || selectedBed.apiId
    || selectedBed.ward?.id
    || selectedBed.ward?.apiId
    || wardInfo?.id
    || wardInfo?.apiId
    || wardInfo?._id
    || null;

  const wardName = selectedBed.wardName
    || selectedBed.ward?.name
    || selectedBed.ward?.wardName
    || selectedBed.department
    || wardInfo?.name
    || wardInfo?.wardName
    || wardInfo?.department
    || 'General Ward';

  const department = selectedBed.department
    || selectedBed.ward?.department
    || selectedBed.ward?.name
    || wardInfo?.department
    || wardInfo?.name
    || 'General';

  return {
    ...selectedBed,
    wardId,
    wardName,
    department,
    apiId: selectedBed.apiId || wardId || null,
    id: selectedBed.id || selectedBed._id || selectedBed.bedId || null,
    bedId: selectedBed.bedId || selectedBed.id || selectedBed._id || null,
    bedNumber: selectedBed.bedNumber || selectedBed.bedId || selectedBed.id || 'Not Assigned'
  };
};

export const buildAssignmentPayload = (selectedBed, wardInfo = {}) => {
  const enrichedBed = enrichSelectedBedWithWard(selectedBed, wardInfo);

  return {
    assignedWardId: enrichedBed?.wardId || null,
    assignedWardName: enrichedBed?.wardName || 'General Ward',
    bedAssigned: enrichedBed?.id || null,
    assignedBedNumber: enrichedBed?.bedNumber || enrichedBed?.id || null
  };
};
